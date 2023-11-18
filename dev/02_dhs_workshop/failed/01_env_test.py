# %%
import os
from pathlib import Path

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

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

weights_dir = get_weight_dir('meta-llama/Llama-2-13b-hf')
model = AutoModelForCausalLM.from_pretrained(weights_dir)
tokenizer = AutoTokenizer.from_pretrained(weights_dir)


# %%
# Let's try embedding some stuff
inputs = tokenizer('Hello Llama', return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}


outputs = model.generate(**inputs, max_length=50, num_beams=5)

print(tokenizer.decode(outputs[0]))







