# Script for converting propaganda data to HF datasets format needed by training script
import datasets
from pathlib import Path

base_dir = Path("/projects/BSTEWART/mj2976/llama-2-training")
data_dir = base_dir / "data"

data_file = data_dir / "prop_articles.csv"


