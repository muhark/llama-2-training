# Coding Notes

31 October 2023

Current attempt: trying to follow blogpost https://huggingface.co/blog/ram-efficient-pytorch-fsdp and https://github.com/pacman100/DHS-LLM-Workshop.

Creating a new directory/environment for this.


## Setup

Copied over relevant code from `DHS-LLM-Workshop/chat_assistant/training`:

- `train.py`
- `utils.py`
- `requirements.txt`
- `llama_flash_attn_monkey_patch.py`

Installing packages

```bash
python -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install torch
pip install packaging
ninja --version; echo $?
pip install flash-attn --no-build-isolation 
pip install -r requirements.txt # Copied 
```

Modified `utils.py` to load offline dataset with `load_from_disk`

Modified `fsdp_config.yaml` to `num_processes: 4`

The following worked:
```yaml
accelerate launch --config_file "configs/fsdp_config.yaml"  train.py \
--model_name $HF_HOME/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235 \
--dataset_name "/scratch/gpfs/mj2976/projects/llama-2-training/data/text_only_split" \
--max_seq_len 2048 \
--max_steps 1000 \
--logging_steps 25 \
--eval_steps 100 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "llama-2-7b-chat_test" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--dataset_text_field "text" \
--num_workers 40 \
--use_gradient_checkpointing \
--learning_rate 5e-5  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--use_flash_attn True \
--push_to_hub=False \
```


### PEFT

Trying

```yaml
python train.py \
--model_name $HF_HOME/models--meta-llama--Llama-2-70b-chat-hf/snapshots/9ff8b00464fc439a64bb374769dec3dd627be1c2 \
--dataset_name "/scratch/gpfs/mj2976/projects/llama-2-training/data/text_only_split" \
--max_seq_len 2048 \
--max_steps 1000 \
--logging_steps 25 \
--eval_steps 200 \
--save_steps 100 \
--push_to_hub False \
--bf16 True \
--packing True \
--output_dir "llama-2-70b-chat_peft_test" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--dataset_text_field "text" \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 32 \
--lora_target_modules "q_proj,k_proj,o_proj,down_proj,up_proj,gate_proj",
--use_4bit_qunatization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--use_flash_attn True
```

Note: had to change `lora_target_modules` to (default) values above. Can be seen by inspecting Llama module