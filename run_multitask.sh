#!/bin/bash
set -x

# model parameters
model_name=llama-3-8b-hf
model_dir=/home/yujin-wa20/projects/aliendao
model_name_full=${model_dir}/${model_name}

# training parameters
data_dir=./dataset
rank=16
lora_alpha=16
lr=3e-4
task=math # or "commonsense" for commonsense reasoning
gradient_accumulation_steps=4
per_device_train_batch_size=4
per_device_eval_batch_size=8
num_train_epochs=3
warmup_ratio=0.1
logging_steps=10
init_lora_weights=qlora
do_train=True
do_eval=True

#! HyCLoRA core parameters
use_hyclora=True
layer_type=intra
iteration_threshold=5
softmax_outlier_ratio=0.05
layernorm_outlier_ratio=0.005
q_bit=4

# tag
tag=${model_name}-${use_hyclora}-${layer_type}-${q_bit}-${layernorm_outlier_ratio}-${softmax_outlier_ratio}
exp_name=wikitext-${tag}

mkdir exp_results_${task}

# command
python -u run_multitask.py \
    --rank $rank \
    --lora_alpha $lora_alpha \
    --lora_init \
    --init_lora_weights $init_lora_weights \
    --model_name_or_path $model_name_full \
    --task $task \
    --data_dir $data_dir \
    --test_split test \
    --use_normalized_template \
    --max_length 512 \
    --seed 42 \
    --learning_rate $lr \
    --max_grad_norm 1 \
    --num_train_epochs $num_train_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --evaluation_strategy no \
    --save_strategy epoch \
    --warmup_ratio $warmup_ratio \
    --greedy_decoding \
    --logging_strategy steps \
    --logging_steps $logging_steps \
    --disable_tqdm false \
    --report_to none \
    --remove_unused_columns false \
    --output_dir exp_results_${task}/${exp_name} \
    --overwrite_output_dir \
    --use_hyclora $use_hyclora \
    --layer_type $layer_type \
    --iteration_threshold $iteration_threshold \
    --softmax_outlier_ratio $softmax_outlier_ratio \
    --layernorm_outlier_ratio $layernorm_outlier_ratio \
    --q_bit $q_bit \
    --do_train $do_train \
    --do_eval $do_eval | tee exp_results_${task}/${exp_name}.log