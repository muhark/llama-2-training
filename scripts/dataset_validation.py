#!/usr/bin/env python
# Checking for data issues

# %%
import os
import json
import datasets
from pathlib import Path
from transformers import AutoTokenizer

base_dir = Path("~/Dev/llama-2-training").expanduser()
data_dir = base_dir / "data"
data_file = data_dir / "prop_articles.json"

# %% Load dataset
ds = datasets.load_from_disk(data_dir/"prop_articles_tokenized_content_cl")


# %% Are there empty rows?
train_text = ds['train']['content_cl']

min(map(len, train_text)) # 60

# %% Maybe some tokenizaton issues?
input_lens = ds['train'].map(
    lambda row: {'input_len': len(row['input_ids'][0])},
    num_proc=14
)['input_len']

input_lens

# %%
min(input_lens)
max(input_lens)

len(ds['train'].select([input_lens.index(max(input_lens))])['content_cl'][0])


# %% Do we run into issues with forward pass?
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

checkpoint = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint, load_in_8bit=True, torch_dtype=torch.bfloat16)

# %% Let's see if we can pass a row
inputs = tokenizer(
    ds['train'].select(range(1))['content_cl'],
    return_tensors="pt"
)

model.generate(**inputs, max_length=60, return_dict=True)
# This works

# %% Let's try the collator
from torch.utils.data import DataLoader
from transformers import default_data_collator

ds.set_format('torch', columns=['input_ids', 'attention_mask'])
train_dataloader = DataLoader(
    ds['train'],
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=2
)

next(iter(train_dataloader))