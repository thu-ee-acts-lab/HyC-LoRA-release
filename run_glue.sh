#!/bin/bash
set -x

#! Model parameters
model_name=roberta-base          # Pre-trained model name
task_name=rte                    # Task name for GLUE benchmark (e.g., RTE for Recognizing Textual Entailment)

# Training parameters
seed=42                          # Random seed for reproducibility
lr=3e-4                          # Learning rate
max_length=128                   # Maximum sequence length for input data
per_device_train_batch_size=32   # Training batch size per device
per_device_eval_batch_size=128   # Evaluation batch size per device
num_train_epochs=10              # Number of training epochs

#! HyCLoRA core parameters
use_hyclora=True                 # Whether to use HyCLoRA
layer_type=intra_inter           # Type of HyCLoRA layer (e.g., intra_inter)
iteration_threshold=5            # Calibration steps for HyCLoRA
softmax_outlier_ratio=0.05       # Outlier ratio for softmax
layernorm_outlier_ratio=0.005    # Outlier ratio for LayerNorm
q_bit=2                          # Quantization bit width

# Experiment tag and output directory
tag=${model_name}-${use_hyclora}-${layer_type}-${q_bit}-${layernorm_outlier_ratio}-${softmax_outlier_ratio}
exp_name=glue-${task_name}-${tag}
output_dir=exp_results_glue/${exp_name}

# Create the output directory
mkdir -p exp_results_glue

# Training command
python -u run_glue.py \
    --model-name-or-path $model_name \
    --task-name $task_name \
    --max-length $max_length \
    --per-device-train-batch-size $per_device_train_batch_size \
    --per-device-eval-batch-size $per_device_eval_batch_size \
    --learning-rate $lr \
    --num-train-epochs $num_train_epochs \
    --seed $seed \
    --output-dir $output_dir \
    --pad-to-max-length \
    --use-hyclora $use_hyclora \
    --layer-type $layer_type \
    --iteration-threshold $iteration_threshold \
    --layernorm-outlier-ratio $layernorm_outlier_ratio \
    --softmax-outlier-ratio $softmax_outlier_ratio \
    --q-bit $q_bit | tee ${output_dir}.log
