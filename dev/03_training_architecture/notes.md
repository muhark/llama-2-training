# Development Notes

## 2023.11.18

Functionality needed for training loop:

- [ ] HuggingFace CausalLM Trainer training script [#7](https://github.com/muhark/llama-2-training/issues/7)
- [ ] PEFT/QLoRA model integration: both for VRAM and storage-on-disk
- [ ] Custom eval callback for trainer - add the eval tasks to this ([#5](https://github.com/muhark/llama-2-training/issues/5)).
- [ ] Dashboard logging for eval tasks - use wandb  ([#6](https://github.com/muhark/llama-2-training/issues/6))

Beginning development on [scripts/train.py](./scripts/train.py) and [scripts/utils.py](scripts/utils.py).

- `train.py` holds the argparser and `main`
- `utils.py` holds all the actual processing steps

Begin basing on the dhs-workshop script in [02_dhs_workshop](../02_dhs_workshop/).

