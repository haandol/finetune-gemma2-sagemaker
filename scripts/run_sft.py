import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

tqdm.pandas()


@dataclass
class ScriptArguments:
    """Represents the script arguments for SFT training."""

    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset, should be /opt/ml/input/data/train_dataset.json"
        },
    )
    model_id: str = field(
        default="google/gemma-2-9b",
        metadata={"help": "Model ID to use for SFT training"},
    )
    use_qlora: bool = field(default=False, metadata={"help": "Whether to use QLORA"})
    merge_adapters: bool = field(
        metadata={"help": "Whether to merge weights for LoRA."},
        default=False,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    # Load dataset from local path
    dataset = load_dataset(
        "json",
        data_files=script_args.dataset_path,
        split="train",
    )

    # on use QLoRA
    if script_args.use_qlora:
        print("Using QLoRA")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        quantization_config = None

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, add_eos_token=False)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",  # gemma2 model can not use Flash-Attention, so it recommends eager mode
        device_map="auto",
    )
    # disable decoding cache for training
    model.config.use_cache = False
    # on enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Peft setup
    # all trainable linear layers
    modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # log number of trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(
        f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%"
    )

    # model training
    tokenizer.padding_side = "right"
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    # save model and tokenizer
    SAGEMAKER_SAVE_DIR = "/opt/ml/model"

    trainer.tokenizer.save_pretrained(SAGEMAKER_SAVE_DIR)

    # on merge LoRA adapter weights with base model
    if script_args.merge_adapters:
        from peft import AutoPeftModelForCausalLM

        # merge adapter weights with base model and save int 4 model
        trainer.model.save_pretrained(training_args.output_dir)
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        # list file in output_dir
        print(os.listdir(training_args.output_dir))

        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(
            SAGEMAKER_SAVE_DIR, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(SAGEMAKER_SAVE_DIR, safe_serialization=True)
