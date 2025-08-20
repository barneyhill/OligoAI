import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

import lightning.pytorch as pl

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanSquaredError

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from rinalmo.data.alphabet import Alphabet
# Assuming your new dataset is in datamodule.py
from rinalmo.data.downstream.aso.datamodule import ASODataModule
from rinalmo.utils.scaler import StandardScaler
from rinalmo.utils.finetune_callback import GradualUnfreezing

class ASOContextInteractionHead(nn.Module):
    """
    Performs cross-attention in a lower-dimensional bottleneck space
    to reduce parameters and prevent overfitting.
    """
    def __init__(
        self,
        c_in: int,          # e.g., 1280 from RiNALMo-giga
        bottleneck_dim: int = 128, # The new, smaller dimension for attention
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Linear layers to project from c_in down to bottleneck_dim
        self.q_proj = nn.Linear(c_in, bottleneck_dim)
        self.k_proj = nn.Linear(c_in, bottleneck_dim)
        self.v_proj = nn.Linear(c_in, bottleneck_dim)

        # 2. Cross-Attention Layer now operates on the smaller dimension
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim, # Using the smaller dimension
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Prediction MLP
        mlp_layers = []
        for i in range(num_layers):
            # Input to MLP is the pooled bottleneck_dim representation + dosage (1)
            in_dim = bottleneck_dim + 1 if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim

            mlp_layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, aso_repr, context_repr, aso_pad_mask, context_pad_mask, dosage):
        # 1. Project ASO (query) and Context (key, value) into bottleneck space
        q = self.q_proj(aso_repr)
        k = self.k_proj(context_repr)
        v = self.v_proj(context_repr)

        # 2. Perform Cross-Attention in the bottleneck space
        context_aware_aso_repr, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=context_pad_mask,
        )

        # 3. Average Pooling over the context-aware ASO representation
        lengths = (~aso_pad_mask).sum(dim=1, keepdim=True).float()
        pooled_repr = context_aware_aso_repr.sum(dim=1) / lengths.clamp(min=1.0)

        # 4. Concatenate with dosage and predict
        combined_input = torch.cat([pooled_repr, dosage.unsqueeze(-1)], dim=-1)
        out = self.mlp(combined_input)
        return out.squeeze(-1)

class ASOInhibitionPredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        num_heads: int = 8, ### NEW: Hyperparameter for attention heads
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scaler = StandardScaler()

        # ### MODIFIED: One shared language model for both ASO and context
        self.lm = RiNALMo(model_config(lm_config))

        # ### MODIFIED: Use the new interaction and prediction head
        self.pred_head = ASOContextInteractionHead(
            c_in=self.lm.config['model']['transformer'].embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.loss = nn.MSELoss()
        self.r2_metric = R2Score()
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError(squared=False)

        self.lr = lr
        self.pad_idx = self.lm.config['model']['embedding'].padding_idx

        self.val_data_by_custom_id = defaultdict(lambda: {'preds': [], 'targets': []})
        self.test_data_by_custom_id = defaultdict(lambda: {'preds': [], 'targets': []})

    def load_pretrained_lm_weights(self, pretrained_weights_path):
        self.lm.load_state_dict(torch.load(pretrained_weights_path))

    ### MODIFIED: The main forward pass of the model
    def forward(self, aso_tokens, context_tokens, dosage):
        # 1. Get embeddings for both ASO and RNA context using the SHARED language model
        aso_repr = self.lm(aso_tokens)["representation"]
        context_repr = self.lm(context_tokens)["representation"]

        # 2. Create padding masks for both sequences
        aso_pad_mask = aso_tokens.eq(self.pad_idx)
        context_pad_mask = context_tokens.eq(self.pad_idx)

        # Nullify padding token representations (good practice)
        aso_repr[aso_pad_mask, :] = 0.0
        context_repr[context_pad_mask, :] = 0.0

        # 3. Pass all representations to the interaction head for prediction
        pred = self.pred_head(aso_repr, context_repr, aso_pad_mask, context_pad_mask, dosage)
        return pred

    def fit_scaler(self, batch):
        # ### MODIFIED: Unpack batch correctly based on new dataset structure
        _, _, inhibition, _, _ = batch
        self.scaler.partial_fit(inhibition)

    def _common_step(self, batch, batch_idx, log_prefix: str):
        # ### MODIFIED: Unpack batch correctly based on new dataset structure
        aso_tokens, context_tokens, inhibition_target, dosage, custom_ids = batch

        # ### MODIFIED: Call the new forward method
        preds = self(aso_tokens, context_tokens, dosage)

        # The rest of this function is unchanged as it deals with metrics and logging
        loss = self.loss(preds, inhibition_target)

        mse = F.mse_loss(preds, inhibition_target)
        mae = F.l1_loss(preds, inhibition_target)

        self.r2_metric.update(preds, inhibition_target)
        self.mae_metric.update(preds, inhibition_target)
        self.rmse_metric.update(preds, inhibition_target)

        log = {
            f'{log_prefix}/loss': loss,
            f'{log_prefix}/mse': mse,
            f'{log_prefix}/mae': mae,
        }
        self.log_dict(log, sync_dist=True)

        return loss, preds, inhibition_target, custom_ids

    # No changes needed from here down to main()
    def _eval_step(self, batch, batch_idx, log_prefix, data_collector):
        loss, preds_percent, inhibition_percent, custom_ids = self._common_step(batch, batch_idx, log_prefix=log_prefix)
        for pred, target, custom_id in zip(preds_percent.cpu().numpy(), inhibition_percent.cpu().numpy(), custom_ids):
            data_collector[custom_id]['preds'].append(pred)
            data_collector[custom_id]['targets'].append(target)
        return loss

    def _on_eval_epoch_start(self):
        self.r2_metric.reset()
        self.mae_metric.reset()
        self.rmse_metric.reset()

    def _calculate_mean_spearman_correlation(self, data_by_custom_id):
        correlations = []
        for custom_id, data in data_by_custom_id.items():
            preds = np.array(data['preds'])
            targets = np.array(data['targets'])
            if len(preds) > 1:
                corr, p_value = spearmanr(preds, targets)
                if not np.isnan(corr):
                    correlations.append(corr)
        if len(correlations) > 0:
            return np.mean(correlations), len(correlations)
        else:
            return 0.0, 0

    def _on_eval_epoch_end(self, log_prefix: str, data_collector):
        if not self.trainer.sanity_checking:
            self.log(f"{log_prefix}/r2", self.r2_metric.compute(), sync_dist=True)
            self.log(f"{log_prefix}/mae_final", self.mae_metric.compute(), sync_dist=True)
            self.log(f"{log_prefix}/rmse", self.rmse_metric.compute(), sync_dist=True)
            mean_spearman, n_custom_ids = self._calculate_mean_spearman_correlation(data_collector)
            self.log(f"{log_prefix}/mean_spearman_corr", mean_spearman, sync_dist=True)
            print(f"{log_prefix} - Mean Spearman correlation: {mean_spearman:.4f} across {n_custom_ids} custom_ids")
            self.r2_metric.reset()
            self.mae_metric.reset()
            self.rmse_metric.reset()
        data_collector.clear()

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            return self.fit_scaler(batch)
        loss, _, _, _ = self._common_step(batch, batch_idx, log_prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="val", data_collector=self.val_data_by_custom_id)

    def on_validation_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end("val", self.val_data_by_custom_id)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="test", data_collector=self.test_data_by_custom_id)

    def on_test_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end("test", self.test_data_by_custom_id)

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=5000)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ### MODIFIED: Pass new `num_heads` argument to the model
    model = ASOInhibitionPredictionWrapper(
        lm_config=args.lm_config,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
    )

    if args.pretrained_rinalmo_weights:
        model.load_pretrained_lm_weights(args.pretrained_rinalmo_weights)

    if args.init_params:
        model.load_state_dict(torch.load(args.init_params))

    alphabet = Alphabet()
    # Your datamodule.py should handle the ASODataset correctly
    datamodule = ASODataModule(
        data_path=args.data_path,
        alphabet=alphabet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed if args.seed else 42,
    )

    callbacks = []
    loggers = []
    if args.wandb:
        wandb_logger = WandbLogger(name=args.wandb_experiment_name, save_dir=args.output_dir, project=args.wandb_project, entity=args.wandb_entity, save_code=True)
        loggers.append(wandb_logger)
    if args.checkpoint_every_epoch:
        callbacks.append(ModelCheckpoint(dirpath=args.output_dir, filename='aso-epoch_ckpt-{epoch}-{step}', every_n_epochs=1, save_top_k=-1))
    callbacks.append(ModelCheckpoint(dirpath=args.output_dir, filename='aso-best-{epoch}-{val/mae_final:.2f}', monitor='val/mae_final', mode='min', save_top_k=1))
    if loggers:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    strategy = 'auto'
    if args.devices != 'auto' and ("," in args.devices or (int(args.devices) > 1 and int(args.devices) != -1)):
        strategy = DDPStrategy(find_unused_parameters=True)
    if args.ft_schedule:
        callbacks.append(GradualUnfreezing(unfreeze_schedule_path=args.ft_schedule))

    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices, max_steps=args.max_steps, max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val, precision=args.precision, default_root_dir=args.output_dir,
        log_every_n_steps=args.log_every_n_steps, strategy=strategy, logger=loggers, callbacks=callbacks,
    )

    if not args.test_only:
        trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (all existing args are fine)
    parser.add_argument("data_path", type=str, help="Path to the CSV file containing ASO inhibition data")
    parser.add_argument("--init_params", type=str, default=None, help="Path to the '.pt' file containing model weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for all output files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False, help="Checkpoint at every epoch")
    parser.add_argument("--test_only", action="store_true", default=False, help="Skip training, run test set")

    # Model
    parser.add_argument("--lm_config", type=str, default="giga", help="Language model configuration")
    parser.add_argument("--pretrained_rinalmo_weights", type=str, default=None, help="Path to pretrained RiNALMo weights")
    # ### NEW: Command line argument for number of attention heads ###
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads for cross-attention")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for MLP head")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in MLP head")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in MLP head")

    # Data split
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of data for validation")

    # W&B
    parser.add_argument("--wandb", action="store_true", default=False, help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_experiment_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (user or team)")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Logging frequency")

    # Data loading
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for CUDA")

    # Training
    parser.add_argument("--ft_schedule", type=str, default=None, help="Fine-tuning schedule file")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accelerator", type=str, default='auto', help="Accelerator")
    parser.add_argument("--devices", type=str, default='auto', help="Devices")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Max training epochs")
    parser.add_argument("--gradient_clip_val", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--precision", type=str, default='16-mixed', help="Training precision")

    args = parser.parse_args()
    main(args)
