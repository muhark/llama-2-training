# Script for converting propaganda data to HF datasets format needed by training script
import json
import datasets
import numpy as np
from pathlib import Path

base_dir = Path("/projects/BSTEWART/mj2976/llama-2-training")
data_dir = base_dir / "data"
data_file = data_dir / "prop_articles.json"
data = json.loads(data_file.read_text().splitlines())
# or
data = [json.loads(record) for record in data_file.read_text().splitlines()]

# Splits
rng = np.random.default_rng(seed=42)






ds = datasets.Dataset.from_json(data_file)

# Do we construct splits at this stage?
# Perhaps yes - 3-way, prevent contamination during training
ds = ds.train_test_split()
