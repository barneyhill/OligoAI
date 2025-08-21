apt-get update; apt-get install vim
pip install torch==2.1.0 flash-attn==2.3.2 -U --force-reinstall
pip install pandas scipy lightning wandb ml_collections
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt # 650M params
mkdir weights
mv rinalmo_giga_pretrained.pt weights

pip uninstall torchvision

python train_aso.py     ./data/aso_inhibitions_13_08_25_incl_context_w_flank_20_df.csv.gz    --pretrained_rinalmo_weights ./weights/rinalmo_giga_pretrained.pt     --output_dir ./aso_finetuning_outputs     --batch_size 256     --lr 5e-4     --max_epochs 50     --hidden_dim 128     --num_layers 3     --dropout 0.1     --wandb     --wandb_project aso_inhibition     --wandb_experiment_name rinalmo_aso_mlp --num_workers 16 --pin_memory --lm_config giga      --ft_schedule ft_schedules/giga_aso_ft.yaml
