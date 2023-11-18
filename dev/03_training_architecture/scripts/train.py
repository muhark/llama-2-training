########################################################################
# Training loop based on example from HuggingFace
# Copied/modified by @muhark
########################################################################

from dataclasses import dataclass, field
import os
import subprocess
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, Trainer
from utils import *

# Script Arguments
@dataclass
class ScriptArguments:
    """
    """
    # Model
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": "Model name in HF repo format or path to weights. If path, must contain `config.json`."
        },
    )

    # Dataset
    dataset_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "The preference dataset to use."},
    )

    # Data preprocessing
    max_seq_length: Optional[int] = field(default=512)
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )

    # Training hyperparams
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: str = field(default="constant")

    # LoRA Config
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_task_type: Optional[str] = field(default="CASUAL_LM")
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )

    # Quantization config
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )


    # Hardware args
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=10, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(default="results", metadata={"help": "Where to store the final model."})
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(default=4, metadata={"help": "Number of dataset workers to use."})
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tests things like proper saving/loading/logging of model"},
    )


def main(args):
    # training arguments
    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true" and args.use_peft_lora
    )
    save_strategy = "no" if is_deepspeed_peft_enabled else "steps"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy=save_strategy,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
        include_tokens_per_second=True,
    )

    # model
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    # trainer
    trainer = Trainer(model=model, args=training_arguments, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.accelerator.print(f"{trainer.model}")
    if args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    if is_deepspeed_peft_enabled:
        trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=args.save_steps))

    if args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, args)

    # train
    trainer.train()

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if is_deepspeed_peft_enabled:
        trainer.accelerator.wait_for_everyone()
        state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        if trainer.accelerator.is_main_process:
            unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        trainer.accelerator.wait_for_everyone()
    else:
        if args.push_to_hub:
            trainer.push_to_hub()
            if args.use_peft_lora:
                trainer.model.push_to_hub(args.output_dir)
        else:
            trainer.save_model(args.output_dir)

    # Save everything else on main process
    if trainer.args.process_index == 0:
        print("Sharding model if >10GB...")
        # FSDP/DeepSpeed save the model as a single `pytorch_model.bin` file, so we need to shard it.
        # We run this in a subprocess to avoid interference from the accelerators.
        subprocess.run(
            [
                "python",
                "shard_checkpoint.py",
                f"--output_dir={args.output_dir}",
            ],
            check=True,
        )
        if "training_args.bin" in os.listdir(args.output_dir):
            os.remove(os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
