#!/bin/bash
set -x

#! Model parameters
model_type=llama                    # Type of the model (e.g., llama)
model_name=llama-2-7b-hf            # Specific model name or path
model_dir=/home/yujin-wa20/projects/aliendao
model_name_full=${model_dir}/${model_name}  # Full path to the model

# Data parameters
seq_len=8192                       # Sequence length for input data
data_dir=./dataset/redpajama_1t_sample_${seq_len}  # Directory containing the dataset
proof_pile_file=proof_pile.bin      # Proof pile file name
pg19_validation_file=pg19_validation.bin  # PG19 validation file name
output_dir=out                      # Directory for saving experiment results
cache_dir=out_cache                 # Directory for caching data or models

#! HyCLoRA core parameters
use_hyclora=True                    # Whether to use HyCLoRA
layer_type=baseline              # Type of HyCLoRA layer (e.g., intra_inter)
iteration_threshold=5               # Calibration steps for HyCLoRA
softmax_outlier_ratio=0.05          # Outlier ratio for softmax
layernorm_outlier_ratio=0.005       # Outlier ratio for LayerNorm
q_bit=4                             # Quantization bit width

# Training parameters
num_train_epochs=1                  # Number of training epochs
per_device_train_batch_size=1       # Training batch size per device
per_device_eval_batch_size=2        # Evaluation batch size per device
gradient_accumulation_steps=8       # Number of gradient accumulation steps
learning_rate=2e-5                  # Learning rate
weight_decay=0.0                    # Weight decay
warmup_steps=20                     # Number of warmup steps
save_steps=200                      # Save model every X steps
save_total_limit=2                  # Maximum number of checkpoints to save
logging_steps=1                     # Log every X steps
max_steps=200                       # Maximum number of training steps
bf16=True                           # Use bf16 precision
use_flash_attn=True                 # Use flash attention
evaluation_strategy="no"            # Evaluation strategy (e.g., "no", "steps")
save_strategy="steps"               # Save strategy (e.g., "steps", "epoch")
lr_scheduler_type="constant_with_warmup"  # Learning rate scheduler type
gradient_checkpointing=False        # Use gradient checkpointing
report_to=none                      # Reporting destination (e.g., "wandb", "tensorboard")

# Experiment tag and output file name
tag=${model_name}-${use_hyclora}-${layer_type}-${q_bit}-${layernorm_outlier_ratio}-${softmax_outlier_ratio}
exp_name=longseq-${tag}
out_name=${exp_name}  # Output file name for logging

# Create output and cache directories
mkdir -p $output_dir
mkdir -p $cache_dir

# Training command
python -u run_longseq.py \
    --model_type $model_type \
    --model_name_or_path $model_name_full \
    --data_dir $data_dir \
    --bf16 $bf16 \
    --output_dir $output_dir/$exp_name \
    --run_name $exp_name \
    --cache_dir $cache_dir \
    --model_max_length $seq_len \
    --use_flash_attn $use_flash_attn \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "$evaluation_strategy" \
    --save_strategy "$save_strategy" \
    --save_steps $save_steps \
    --save_total_limit $save_total_limit \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type "$lr_scheduler_type" \
    --logging_steps $logging_steps \
    --report_to $report_to \
    --gradient_checkpointing $gradient_checkpointing \
    --max_steps $max_steps \
    --use_hyclora $use_hyclora \
    --layer_type $layer_type \
    --iteration_threshold $iteration_threshold \
    --layernorm_outlier_ratio $layernorm_outlier_ratio \
    --softmax_outlier_ratio $softmax_outlier_ratio \
    --q_bit $q_bit | tee $output_dir/${out_name}.log

