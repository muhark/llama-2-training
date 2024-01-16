#!/bin/bash

# Quick script for running CLM using script on Della

# Assuming environment is set up

BASE_DIR=/projects/BSTEWART/mj2976
SCRIPT_DIR

### ARGS ###
dataset=${DATASET:-$BASE_DIR/llama-2-training/data/prop_articles_tokenized_content_cl}
model=${MODEL:-/scratch/gpfs/mj2976/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/638c8be6b16b6cb237274a65392c1045f7c4132c}


lr=${LR:-6e-4}
bsz=${BSZ:-32} # batch size (training)
wd=${WD:-0.01} # weight decay
epochs=${EPOCHS:-1}
warmup=${WARMUP:-0.1}
seed=${SEED:-42}


# output dirs
suffix=${SUFFIX:-}
output_dir=${OUTPUT_DIR:-checkpoints/run_clm_testing}${suffix}

header="accelerate run $SCRIPT_DIR/run_clm_hpc.py"

base_arguments=(
    ### Dataset args ###
    --dataset_name $dataset
    # --train_file TRAIN_FILE 
    # --validation_file VALIDATION_FILE
    # --validation_split_percentage VALIDATION_SPLIT_PERCENTAGE

    ### Model args ###
    --model_name_or_path $model
    # --config_name $model
    # --tokenizer_name $model
    # --use_slow_tokenizer true/false

    ### Training args ###
    --per_device_train_batch_size $bsz
    --per_device_eval_batch_size $bsz
    --learning_rate $lr
    --weight_decay $wd
    --num_train_epochs $epochs
    # --max_train_steps MAX_TRAIN_STEPS
    --gradient_accumulation_steps $(expr $bsz / $SEQ_DATA_PARALLEL_SIZE)
    --lr_scheduler_type cosine
    --num_warmup_steps 2000

    ### Output args ###
    --output_dir $output_dir
    --seed $seed
    # --model_type llama # only for training from scratch

    ### Preprocessing args ###
    --block_size $block_size
    --preprocessing_num_workers 12
    --overwrite_cache false
    # --no_keep_linebreaks  Do not keep line breaks when using TXT files.
    --push_to_hub false
    # --hub_model_id 
    # --hub_token HUB_TOKEN
    --trust_remote_code true
    --checkpointing_steps 1000
    --resume_from_checkpoint true
    --with_tracking true
    --report_to wandb
    --low_cpu_mem_usage true
    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out

