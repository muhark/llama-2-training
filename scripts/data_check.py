#!/usr/bin/env python
# Script for converting propaganda data to HF datasets format needed by training script
import os
import json
import datasets
from pathlib import Path
from transformers import AutoTokenizer

base_dir = Path("/projects/BSTEWART/mj2976/llama-2-training")
data_dir = base_dir / "data"
data_file = data_dir / "prop_articles.json"

dataset = datasets.load_from_disk(str(data_dir/"prop_articles_tokenized_content_cl"))

