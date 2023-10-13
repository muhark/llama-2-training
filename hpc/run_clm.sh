#!/bin/bash

# Quick script for running CLM using script on Della

# Assuming environment is set up
BASE_DIR=/scratch/gpfs/mj2976/projects/llama-2-training
SCRIPT_DIR=$BASE_DIR/hpc
DATA_DIR=$BASE_DIR/data
RUN_DIR=${RUN_DIR:-$BASE_DIR/checkpoints}

### ARGS ###
# Model
model=${MODEL:-/scratch/gpfs/mj2976/.cache/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235}

# Data
dataset=${DATASET:-$DATA_DIR/text_only_split}
# train_file=${TRAIN_FILE:-$DATA_DIR/text_trainvalid_split.json} # train includes train and valid, using held-out test
# validation_split_percentage=${VALIDATION_SPLIT_PERCENTAGE:-0.1} # between 0 and 1
block_size=${BLOCK_SIZE:-4096} # 4096 possible?

# Training
lr=${LR:-6e-4}
bsz=${BSZ:-2} # batch size (training)
wd=${WD:-0.01} # weight decay
epochs=${EPOCHS:-1}
warmup=${WARMUP:-0.1}
seed=${SEED:-42}

# Multi-device config
num_cpus=${NUM_CPUS:-10} # per device
num_gpus=$(nvidia-smi -L | wc -l)
gradient_accumulation_steps=1
accelerate_config=${ACCELERATE_CONFIG:-$SCRIPT_DIR/accelerate_config.yaml}

# output dirs
suffix=${SUFFIX:-}
output_dir=${OUTPUT_DIR:-$RUN_DIR/run_clm_testing}${suffix}
mkdir -p $output_dir

# Scripting
header="accelerate launch \
    --config_file ${accelerate_config} \
    $SCRIPT_DIR/run_clm_hpc.py"
base_arguments=(
    ### Dataset args ###
    --dataset_path $dataset
    # --train_file $train_file
    # --validation_split_percentage $validation_split_percentage

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
    --gradient_accumulation_steps $gradient_accumulation_steps
    --lr_scheduler_type cosine
    --num_warmup_steps 2000

    ### Output args ###
    --output_dir $output_dir
    --seed $seed
    # --model_type llama # only for training from scratch

    ### Preprocessing args ###
    --block_size $block_size
    --preprocessing_num_workers $(expr $num_cpus - 1)
    # --overwrite_cache false
    # --no_keep_linebreaks  Do not keep line breaks when using TXT files.
    # --push_to_hub false
    # --hub_model_id 
    # --hub_token HUB_TOKEN
    # --trust_remote_code
    --checkpointing_steps 1000
    # --resume_from_checkpoint $output_dir
    # --with_tracking
    # --report_to wandb
    # --low_cpu_mem_usage
    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" #2>&1 | tee -a $output_dir/log.out

