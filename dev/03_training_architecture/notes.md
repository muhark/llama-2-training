# Development Notes

## 2023.11.18

Functionality needed for training loop:

- [ ] HuggingFace CausalLM Trainer training script [#7](https://github.com/muhark/llama-2-training/issues/7)
- [ ] PEFT/QLoRA model integration: both for VRAM and storage-on-disk
- [ ] Custom eval callback for trainer - add the eval tasks to this ([#5](https://github.com/muhark/llama-2-training/issues/5)).
- [ ] Dashboard logging for eval tasks - use wandb  ([#6](https://github.com/muhark/llama-2-training/issues/6))

Beginning development in [scripts](./scripts).

- `train.py` and `utils.py` are based on the [previous iteration](../02_dhs_workshop/).
- `sft.py` based on [sft.py from trl](https://github.com/huggingface/trl/blob/28bdb6a3736b09f8bad7961f16852d052c74ae04/examples/scripts/sft.py)

[`sft.py`](./scripts/sft.py) seems to be promising for having the necessary tools.

NB: Model saving/loading [here](https://huggingface.co/docs/peft/v0.6.2/en/quicktour#save-and-load-a-model), tldr there is an `AutoPeftModelForCausalLM.from_pretrained` convenience loader.


Still to do:

- Check that added args line up. (Maybe diff with original script?)
- Write custom eval loop and add as callback in trainer.
- Figure out wandb logging (probably changed since I last looked at it)
