import os
import random
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl.commands.cli_utils import TrlParser
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
    parser = TrlParser((ScriptArguments, SFTConfig))
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
    if training_args.distributed_state.is_main_process:
        for index in random.sample(range(len(train_dataset)), 2):
            print("sample training prompt: ", train_dataset[index]["prompt"])
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

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
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
    trainer.train()
    # save adapter weights
    trainer.save_model(training_args.output_dir)

    # save model and tokenizer
    SAGEMAKER_OUTPUT_DIR = "/opt/ml/model"
    if training_args.distributed_state.is_main_process:
        trainer.tokenizer.save_pretrained(SAGEMAKER_OUTPUT_DIR)

    del model
    del trainer
    torch.cuda.empty_cache()

    if training_args.distributed_state.is_main_process:
        merge_and_save_model(
            script_args.model_id,
            training_args.output_dir,
            SAGEMAKER_OUTPUT_DIR,
        )
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print
