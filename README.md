# OligoAI

```python
bash install.sh
python3 train_aso.py ./data/aso_inhibitions_21_08_25_incl_context_w_flank_50_df.csv.gz \
            --pretrained_rinalmo_weights /workspace/rinalmo_giga_pretrained.pt \
            --output_dir ./aso_finetuning_outputs_frozen \
            --batch_size 64 \
            --lr 5e-5 \
            --max_epochs 10 \
            --hidden_dim 128 \
            --num_layers 3 \
            --dropout 0.3 \
            --gradient_clip_val 0.5 \
            --wandb --wandb_project aso_inhibition \
            --wandb_experiment_name rinalmo_frozen_high_reg \
            --num_workers 16 \
            --pin_memory \
            --lm_config giga \
            --checkpoint_every_epoch \
            --ft_schedule ft_schedules/giga_sec_struct_ft.yaml \
            --seed 1
```
