# Script for converting propaganda data to HF datasets format needed by training script
import os
import json
import datasets
from pathlib import Path
from transformers import AutoTokenizer

base_dir = Path("/projects/BSTEWART/mj2976/llama-2-training")
data_dir = base_dir / "data"
data_file = data_dir / "prop_articles.json"
data = json.loads(data_file.read_text().splitlines())
# or
data = [json.loads(record) for record in data_file.read_text().splitlines()]

# Initial construction
ds = datasets.Dataset.from_json(str(data_file))
ds.save_to_disk(str(data_dir/"prop_articles_nosplit"))

# Create splits
train_testvalid = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
train_test_valid_ds = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']
})
train_test_valid_ds.save_to_disk(str(data_dir/"prop_articles"))

# Tokenize data
def get_weight_dir(
        model_ref: str,
        hf_cache_dir: Path=Path(os.environ.get("HF_HOME")),
        revision: str='main'
        ) -> Path:
    """
    Convenience function for retrieving locally stored HF weights.
    """
    if not isinstance(hf_cache_dir, Path):
        hf_cache_dir = Path(hf_cache_dir)
    model_path = "--".join(['models'] + model_ref.split('/'))
    snapshot = (hf_cache_dir / f'{model_path}/refs/{revision}').read_text()
    model_weights_dir = hf_cache_dir / f"{model_path}/snapshots/{snapshot}"
    return model_weights_dir

weights_dir = get_weight_dir('meta-llama/Llama-2-13b-hf')
tokenizer = AutoTokenizer.from_pretrained(weights_dir)

# Tokenize data
train_test_valid_ds = datasets.load_from_disk(str(data_dir/"prop_articles"))
train_test_valid_ds = train_test_valid_ds.map(
    lambda batch: tokenizer(batch['content_cl'],
                            return_tensors='pt'),
    batched=False,
    num_proc=8)
train_test_valid_ds.save_to_disk(str(data_dir/"prop_articles_tokenized_content_cl"))

