# Experiment 1 — upper_bound (LoRA, no DP)
torchrun --standalone --nproc_per_node=4 dp_moe_olmoe.py \
  --data_dir /home/jovyan/gpus-4-nodes-volume/duc/data/GLUE-SST-2 \
  --output_dir ./dist_out/upper_bound --experiment upper_bound \
  --lora_r 8 --lora_alpha 16 --finetune_epochs 20 \
  --train_batch_size 1024 --micro_batch_size 128 \
  --wandb_project dp-moe-olmoe --wandb_name upper_bound_run1 --show_progress

# Experiment 2 — upper_bound_no_lora (full weights, no DP)
torchrun --standalone --nproc_per_node=4 dp_moe_olmoe.py \
  --data_dir /home/jovyan/gpus-4-nodes-volume/duc/data/GLUE-SST-2 \
  --output_dir ./dist_out/upper_bound_no_lora --experiment upper_bound_no_lora \
  --finetune_epochs 20 \
  --train_batch_size 1024 --micro_batch_size 16 \
  --wandb_project dp-moe-olmoe --wandb_name upper_bound_no_lora_run1 --show_progress

# Experiment 3 — naive_dp (LoRA + token routing)
torchrun --standalone --nproc_per_node=4 dp_moe_olmoe.py \
  --data_dir /home/jovyan/gpus-4-nodes-volume/duc/data/GLUE-SST-2 \
  --output_dir ./dist_out/naive_dp --experiment naive_dp \
  --epsilon_total 8.0 --delta -1 --lora_r 8 --lora_alpha 16 --finetune_epochs 20 \
  --train_batch_size 1024 --micro_batch_size 32 \
  --wandb_project dp-moe-olmoe --wandb_name naive_dp_eps8 --show_progress

# Experiment 4 — naive_dp_no_lora (full weights, Tholoniat 2024 exact)
torchrun --standalone --nproc_per_node=4 dp_moe_olmoe.py \
  --data_dir /home/jovyan/gpus-4-nodes-volume/duc/data/GLUE-SST-2 \
  --output_dir ./dist_out/naive_dp_no_lora --experiment naive_dp_no_lora \
  --epsilon_total 8.0 --delta -1 --finetune_epochs 20 \
  --train_batch_size 1024 --micro_batch_size 8 \
  --wandb_project dp-moe-olmoe --wandb_name naive_dp_no_lora_eps8 --show_progress

# Experiment 5 — ours (record routing + parallel composition)
torchrun --standalone --nproc_per_node=4 dp_moe_olmoe.py \
  --data_dir /home/jovyan/gpus-4-nodes-volume/duc/data/GLUE-SST-2 \
  --output_dir ./dist_out/ours --experiment ours \
  --epsilon_total 8.0 --epsilon_attention 4.0 --delta -1 \
  --num_layers_to_train 4 --attention_epochs 8 --expert_epochs 12 \
  --lora_r 8 --lora_alpha 16 --train_batch_size 1024 --micro_batch_size 128 \
  --wandb_project dp-moe-olmoe --wandb_name ours_eps8 \
  --eval_each_layer --show_progress


python /home/quyennd/Data_HDD/ducdm/ducb200/dp_moe_rea.py --data_dir /home/quyennd/Data_HDD/ducdm/ducb200/data/GLUE-SST-2 --output_dir /home/quyennd/Data_HDD/ducdm/ducb200/results_switch --no_dp --show_progress