#!/bin/bash
set -x

#! model parameters
model_name=llama-2-7b-hf
model_dir=/home/yujin-wa20/projects/aliendao
model_name_full=${model_dir}/${model_name}

#! HyCLoRA core parameters
use_hyclora=True
layer_type=intra
iteration_threshold=5
softmax_outlier_ratio=0.05
layernorm_outlier_ratio=0
q_bit=8

# training parameters
lora_init_type=qlora
lr=3e-4
lora_rank=16
lora_alpha=16
gradient_accumulation_steps=4
num_train_epochs=6
per_device_train_batch_size=4
evaluation_strategy="no"
save_strategy="no"
weight_decay=0.1
warmup_ratio=0.03
lr_scheduler_type="cosine"
logging_steps=10
do_train=True
do_eval=True

# tag
tag=${model_name}-${use_hyclora}-${layer_type}-${q_bit}-${layernorm_outlier_ratio}-${softmax_outlier_ratio}
exp_name=gsm8k-${tag}-allquant

mkdir exp_results_gsm8k

# command
python -u run_gsm8k.py \
    --lora_init \
    --init_lora_weights $lora_init_type \
    --model_name_or_path $model_name_full \
    --data_name gsm8k \
    --learning_rate $lr \
    --rank $lora_rank \
    --lora_alpha $lora_alpha \
    --seed 11 \
    --expt_name $exp_name \
    --output_dir exp_results_gsm8k/$exp_name/ \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy $evaluation_strategy \
    --save_strategy $save_strategy \
    --weight_decay $weight_decay \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type \
    --logging_steps $logging_steps \
    --do_train $do_train \
    --softmax_outlier_ratio $softmax_outlier_ratio \
    --use_hyclora $use_hyclora \
    --layer_type $layer_type \
    --iteration_threshold $iteration_threshold \
    --layernorm_outlier_ratio $layernorm_outlier_ratio \
    --q_bit $q_bit \
    --report_to none | tee exp_results_gsm8k/$exp_name.log

echo $tag