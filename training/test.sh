#!/bin/bash -l
#SBATCH -J train
#SBATCH -N 1 -n 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=190G
#SBATCH -t 0-24:00:00

# Adapted from test_llama_4_nodes.sh from CodeCreator

module purge
module load anaconda3/2023.3 cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0

BASE_DIR=/projects/BSTEWART/mj2976
LLAMAO_DIR=$BASE_DIR/Llamao
cd LLAMAO_DIR

conda activate llamao

# Training vars
model=${MODEL:-meta-llama/Llama-2-70b-hf}
lr=${LR:-6e-4}
bsz=${BSZ:-32}
warmup=${WARMUP:-0.1}
suffix=${SUFFIX:-}
epochs=${EPOCHS:-1}
dataset=${DATASET:-$BASE_DIR/llama-2-training/data/prop_articles_tokenized_content_cl}

# Output dir
run_name="${model////--}_bsz${bsz}_lr${lr}_epochs${epochs}_warmup${warmup}_$(basename ${dataset})${suffix}"
out_dir="checkpoints/$run_name"
mkdir -p $out_dir

# Cache dir
local_cache=/tmp/$USER/llama-2-training
cache_dir=${CACHE_DIR:-$local_cache}
mkdir -p $cache_dir

# GPU configs
nvidia-smi
num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
num_gpus=${NUM_GPUS:-$num_gpus}

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$num_gpus
export WANDB_PROJECT="llama-2-training"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"

WORLD_SIZE=4 # Total number of GPUs
SEQ_PARALLEL_SIZE=1 # Do not split across gpus
SEQ_DATA_PARALLEL_SIZE=$(expr $WORLD_SIZE / $SEQ_PARALLEL_SIZE)

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# free_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# export MASTER_PORT=$free_port

free_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

header="torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:$free_port\
--nnodes=1 \
--nproc-per-node=$num_gpus \
train.py"

export FSDP_SHARDING_STRATEGY="1" # 5 corresponds to _hybrid_shard_zero2
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"

base_arguments=(
    --report_to wandb
    --do_train
    --model_name_or_path $model
    --tokenizer_name togethercomputer/LLaMA-2-7B-32K
    --run_name $run_name
    --output_dir $out_dir
    --gradient_accumulation_steps $(expr $bsz / $SEQ_DATA_PARALLEL_SIZE)
    --per_device_train_batch_size 1 # We fit the longest sequence per device
    --learning_rate $lr
    --warmup_ratio $warmup
    --seed 42
    --logging_nan_inf_filter false

    --save_steps 50
    --logging_steps 1
    --logging_strategy steps
    --log_time_interval 0
    --log_level info
    --optim adamw_torch

    --bf16
    --num_train_epochs $epochs
    --dataloader_num_workers 4
    --cache_dir $cache_dir
    --overwrite_output_dir
    --disable_tqdm true
    --ddp_find_unused_parameters false

    --fsdp auto_wrap

    --tokenized_train_dataset $dataset/train
    --lora
    --lora_r 16
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target_modules q_proj v_proj o_proj k_proj
    --use_fast_tokenizer false
    --seq_parallel_size $SEQ_PARALLEL_SIZE
    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out
