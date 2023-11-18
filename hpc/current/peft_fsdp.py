# Quick script for peft + fsdp training
import os
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
# from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset


# Default args first
args = Namespace()

# Model directory
hf_model_dir = os.environ.get("HF_HOME", './models')
# args.model_path = hf_model_dir + '/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235'
args.model_path = hf_model_dir + "/models--meta-llama--Llama-2-70b-chat-hf/snapshots/9ff8b00464fc439a64bb374769dec3dd627be1c2"

# bnb config
args.use_4bit_qunatization = True
args.bnb_4bit_quant_type = "nf4"
args.bnb_4bit_compute_dtype = "bfloat16"
compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
args.use_nested_quant = True

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit_qunatization,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
)


# PEFT Config
args.lora_alpha=32
args.lora_r=8
args.lora_dropout=0.1
args.lora_target_modules="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
lora_config = dict(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=args.lora_target_modules.split(","),
)
# Auto device map?
device_map = 'auto'

# Load
args.use_gradient_checkpointing = True
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    load_in_8bit=False,
    quantization_config=bnb_config,
    device_map=device_map,
    use_cache=not args.use_gradient_checkpointing,
    trust_remote_code=True,
    use_flash_attention_2=args.use_flash_attn,
)





# Load things
# accelerator = Accelerator()
peft_config = LoraConfig(**lora_config)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='balanced')
model = get_peft_model(model, peft_config)


# Dataset
args.dataset_path = "/scratch/gpfs/mj2976/projects/llama-2-training/data/blocks_1024" 
dataset = DatasetDict.load_from_disk(args.dataset_path)
dataset.set_format('torch')
ds = dataset['train'].select(range(1000))
batch = ds[range(4)]


# Forward / backward
model.train()
outputs = model(**batch)
loss = outputs.loss
loss.backward()




