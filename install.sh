apt-get update; apt-get install vim
pip install .
pip install flash-attn==2.3.2 -U --force-reinstall
pip install pandas scipy
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt # 650M params
mkdir weights
mv rinalmo_giga_pretrained.pt weights

python train_aso.py     data/aso_inhibitions_13_08_25_incl_metadata_df.csv    --pretrained_rinalmo_weights weights/rinalmo_giga_pretrained.pt     --output_dir ./aso_finetuning_outputs     --batch_size 64     --lr 1e-4     --max_epochs 50     --hidden_dim 128     --num_layers 3     --dropout 0.1     --wandb     --wandb_project aso_inhibition     --wandb_experiment_name rinalmo_aso_mlp --num_workers 16 --pin_memory --lm_config giga      --ft_schedule ft_schedules/giga_aso_ft.yaml
