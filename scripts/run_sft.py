import os
import random
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

tqdm.pandas()


@dataclass
class ScriptArguments:
    """Represents the script arguments for SFT training."""

    train_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset, should be /opt/ml/input/data/training/train_dataset.json"
        },
    )
    test_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset, should be /opt/ml/input/data/training/test_dataset.json"
        },
    )
    model_id: str = field(
        default="google/gemma-2-9b",
        metadata={"help": "Model ID to use for SFT training"},
    )


def merge_and_save_model(model_id: str, adapter_dir: str, output_dir: str) -> None:
    print("Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving the newly created merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    # set seed
    set_seed(training_args.seed)

    # Load dataset from local path
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.test_dataset_path),
        split="train",
    )
    # print random sample on rank 0
    for index in random.sample(range(len(train_dataset)), 2):
        print("sample training prompt: ", train_dataset[index]["prompt"])

    # only use 4-bit quantization for the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, add_eos_token=False)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",  # gemma2 model can not use Flash-Attention
        device_map="auto",
        force_download=True,
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

    # model training
    tokenizer.padding_side = "right"
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        dataset_text_field=training_args.dataset_text_field,
        packing=training_args.packing,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    # train
    trainer.model.print_trainable_parameters()
    trainer.train()
    # save adapter weights
    trainer.save_model()

    del model
    del trainer
    torch.cuda.empty_cache()

    # save model and tokenizer
    SAGEMAKER_SAVE_DIR = "/opt/ml/model"
    merge_and_save_model(
        script_args.model_id, trainer.model.adapter_dir, SAGEMAKER_SAVE_DIR
    )
    trainer.tokenizer.save_pretrained(SAGEMAKER_SAVE_DIR)
