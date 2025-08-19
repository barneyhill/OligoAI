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

from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.aso.datamodule import ASODataModule
from rinalmo.utils.scaler import StandardScaler
from rinalmo.utils.finetune_callback import GradualUnfreezing


class ASOInhibitionHead(nn.Module):
    """Simple MLP head for ASO inhibition prediction"""
    def __init__(
        self,
        c_in: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = c_in if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:  # Don't add activation/dropout after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, pad_mask):
        # Average pooling over sequence length (excluding padding)
        lengths = (~pad_mask).sum(dim=1, keepdim=True).float()
        x_sum = x.sum(dim=1)
        x_mean = x_sum / lengths.clamp(min=1.0)
        
        # Pass through MLP
        out = self.mlp(x_mean)
        return out.squeeze(-1)


class ASOInhibitionPredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.scaler = StandardScaler()
        
        self.lm = RiNALMo(model_config(lm_config))
        
        self.pred_head = ASOInhibitionHead(
            c_in=self.lm.config['model']['transformer'].embed_dim,
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
    
    def load_pretrained_lm_weights(self, pretrained_weights_path):
        self.lm.load_state_dict(torch.load(pretrained_weights_path))
    
    def forward(self, tokens):
        x = self.lm(tokens)["representation"]
        
        # Nullify padding token representations
        pad_mask = tokens.eq(self.pad_idx)
        x[pad_mask, :] = 0.0
        
        pred = self.pred_head(x, pad_mask)
        return pred
    
    def fit_scaler(self, batch):
        _, inhibition = batch
        self.scaler.partial_fit(inhibition)
    
    def _common_step(self, batch, batch_idx, log_prefix: str):
        seq_encoded, inhibition_target = batch
        preds = self(seq_encoded)
        
        # Scale targets and compute loss
        scaled_inhibition_target = self.scaler.transform(inhibition_target)
        loss = self.loss(preds, scaled_inhibition_target)
        
        # Unscale predictions for metrics (back to 0-1 range)
        preds_unscaled = self.scaler.inverse_transform(preds).clamp(min=0.0, max=1.0)
        
        # Convert to percentage for interpretable metrics
        preds_percent = preds_unscaled * 100
        inhibition_percent = inhibition_target * 100
        
        mse = F.mse_loss(preds_percent, inhibition_percent)
        mae = F.l1_loss(preds_percent, inhibition_percent)
        
        # Update metrics
        self.r2_metric.update(preds_percent, inhibition_percent)
        self.mae_metric.update(preds_percent, inhibition_percent)
        self.rmse_metric.update(preds_percent, inhibition_percent)
        
        log = {
            f'{log_prefix}/loss': loss,
            f'{log_prefix}/mse': mse,
            f'{log_prefix}/mae': mae,
        }
        self.log_dict(log, sync_dist=True)
        
        return loss
    
    def _eval_step(self, batch, batch_idx, log_prefix):
        return self._common_step(batch, batch_idx, log_prefix=log_prefix)
    
    def _on_eval_epoch_start(self):
        # Reset metric calculators
        self.r2_metric.reset()
        self.mae_metric.reset()
        self.rmse_metric.reset()
    
    def _on_eval_epoch_end(self, log_prefix: str):
        # Log and reset metric calculators
        if not self.trainer.sanity_checking:
            self.log(f"{log_prefix}/r2", self.r2_metric.compute(), sync_dist=True)
            self.log(f"{log_prefix}/mae_final", self.mae_metric.compute(), sync_dist=True)
            self.log(f"{log_prefix}/rmse", self.rmse_metric.compute(), sync_dist=True)
            
            self.r2_metric.reset()
            self.mae_metric.reset()
            self.rmse_metric.reset()
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            return self.fit_scaler(batch)
        
        return self._common_step(batch, batch_idx, log_prefix="train")
    
    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="val")
    
    def on_validation_epoch_start(self):
        return self._on_eval_epoch_start()
    
    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end("val")
    
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="test")
    
    def on_test_epoch_start(self):
        return self._on_eval_epoch_start()
    
    def on_test_epoch_end(self):
        return self._on_eval_epoch_end("test")
    
    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=5000)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


def main(args):
    if args.seed:
        pl.seed_everything(args.seed)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Model
    model = ASOInhibitionPredictionWrapper(
        lm_config=args.lm_config,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
    )
    
    if args.pretrained_rinalmo_weights:
        model.load_pretrained_lm_weights(args.pretrained_rinalmo_weights)
    
    if args.init_params:
        model.load_state_dict(torch.load(args.init_params))
    
    # Datamodule
    alphabet = Alphabet()
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
    
    # Set up callbacks and loggers
    callbacks = []
    loggers = []
    
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.wandb_experiment_name,
            save_dir=args.output_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_code=True,
        )
        loggers.append(wandb_logger)
    
    if args.checkpoint_every_epoch:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='aso-epoch_ckpt-{epoch}-{step}',
            every_n_epochs=1,
            save_top_k=-1
        )
        callbacks.append(epoch_ckpt_callback)
    
    # Add best model checkpoint callback
    best_ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='aso-best-{epoch}-{val/mae_final:.2f}',
        monitor='val/mae_final',
        mode='min',
        save_top_k=1,
    )
    callbacks.append(best_ckpt_callback)
    
    if loggers:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # Training
    strategy = 'auto'
    if args.devices != 'auto' and ("," in args.devices or (int(args.devices) > 1 and int(args.devices) != -1)):
        strategy = DDPStrategy(find_unused_parameters=True)
    
    if args.ft_schedule:
        ft_callback = GradualUnfreezing(
            unfreeze_schedule_path=args.ft_schedule,
        )
        callbacks.append(ft_callback)
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_every_n_steps,
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
    )
    
    if not args.test_only:
        trainer.fit(model=model, datamodule=datamodule)
    
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "data_path", type=str,
        help="Path to the pickle file containing ASO inhibition data"
    )
    parser.add_argument(
        "--init_params", type=str, default=None,
        help="Path to the '.pt' file containing model weights to use as starting point"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for all output files (checkpoints, logs, etc.)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="Whether to checkpoint at the end of every training epoch"
    )
    parser.add_argument(
        "--test_only", action="store_true", default=False,
        help="Skip training and only run evaluation on test set"
    )
    
    # Model
    parser.add_argument(
        "--lm_config", type=str, default="giga",
        help="Language model configuration"
    )
    parser.add_argument(
        "--pretrained_rinalmo_weights", type=str, default=None,
        help="Path to the pretrained RiNALMo model weights"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128,
        help="Hidden dimension for MLP head"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3,
        help="Number of layers in MLP head"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate in MLP head"
    )
    
    # Data split
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Proportion of data for validation"
    )
    
    # W&B
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_experiment_name", type=str, default=None,
        help="Name of the current experiment for wandb logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="Wandb username or team name"
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50,
        help="How often to log within steps"
    )
    
    # Data loading
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help="Pin memory for CUDA data loading"
    )
    
    # Training
    parser.add_argument(
        "--ft_schedule", type=str, default=None,
        help="Path to the fine-tuning schedule file"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--accelerator", type=str, default='auto',
        help="Accelerator type (cpu, gpu, tpu, etc.)"
    )
    parser.add_argument(
        "--devices", type=str, default='auto',
        help="Devices to use for training"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=-1,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=None,
        help="Gradient clipping value"
    )
    parser.add_argument(
        "--precision", type=str, default='16-mixed',
        help="Training precision"
    )
    
    args = parser.parse_args()
    main(args)
