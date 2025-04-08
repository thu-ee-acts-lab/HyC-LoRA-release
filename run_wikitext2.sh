#!/bin/bash
set -x

# model parameters
model_name=llama-3-8b-hf
model_dir=/home/yujin-wa20/projects/aliendao
model_name_full=${model_dir}/${model_name}

# training parameters
lora_init=True
lora_init_type=qlora
attn_implementation=eager
rank=16
lora_alpha=16
lr=3e-4
gradient_accumulation_steps=128
num_train_epochs=3
per_device_train_batch_size=2
per_device_eval_batch_size=4
save_strategy=epoch
weight_decay=0.1
warmup_ratio=0.03
lr_scheduler_type=cosine
logging_steps=1
do_train=True
do_eval=True
block_size=1024

#! HyCLoRA core parameters
use_hyclora=True
layer_type=intra_inter_full_fuse
iteration_threshold=5
softmax_outlier_ratio=0.05
layernorm_outlier_ratio=0.005
q_bit=2

# tag
tag=${model_name}-${use_hyclora}-${layer_type}-${q_bit}-${layernorm_outlier_ratio}-${softmax_outlier_ratio}
exp_name=wikitext-${tag}

mkdir exp_results_wikitext

# command
python -u run_wikitext2.py \
    --model_name_or_path ${model_name_full} \
    --lora_init ${lora_init} \
    --lora_init_type ${lora_init_type} \
    --attn_implementation ${attn_implementation} \
    --rank ${rank} \
    --lora_alpha ${lora_alpha} \
    --use_hyclora ${use_hyclora} \
    --layer_type ${layer_type} \
    --iteration_threshold ${iteration_threshold} \
    --softmax_outlier_ratio ${softmax_outlier_ratio} \
    --layernorm_outlier_ratio ${layernorm_outlier_ratio} \
    --q_bit ${q_bit} \
    --output_dir exp_results_wikitext/${exp_name}/ \
    --learning_rate ${lr} \
    --seed 11 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --save_strategy ${save_strategy} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps ${logging_steps} \
    --do_train ${do_train} \
    --do_eval ${do_eval} \
    --block_size ${block_size} \
    --report_to none | tee exp_results_wikitext/${exp_name}.log

echo $tag