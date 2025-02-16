import torch
import os
import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from models import TrainingExample, TrainingCheckpoint

# Authenticate Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Hugging Face token not found! Set HUGGINGFACE_TOKEN in env.")

login(token=HF_TOKEN)

def prepare_datasets():
    """Prepare datasets with multi-turn context and sales flow."""
    print("\n=== Preparing Datasets ===")

    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        examples = session.query(TrainingExample).all()
        print(f"Loaded {len(examples)} examples from database")

        all_examples = []
        system_prompt = """You are an elite VIP CEO-level sales representative.
        Engage naturally, identify needs, present solutions, handle objections, guide to checkout, and follow up for retention."""

        for ex in examples:
            input_text = f"{system_prompt}\n\nCustomer: {ex.customer_message}"
            if ex.meta_info:
                input_text += f"\n[CRM Data: {ex.meta_info}]"

            example = {
                "input": input_text,
                "output": f"CEO Salesman: {ex.agent_response}"
            }
            all_examples.append(example)

        dataset = Dataset.from_list(all_examples)
        splits = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        
        return splits["train"], splits["test"]
    except Exception as e:
        print(f"Error preparing datasets: {e}")
        return None, None

def prepare_for_lora():
    print("\n=== Preparing for LoRA Training on 5x A100s ===")

    try:
        # Load model using Hugging Face authentication
        print("1. Loading Mistral-7B...")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            token=HF_TOKEN,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            use_flash_attention_2=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            token=HF_TOKEN
        )

        # Prepare datasets
        print("\n2. Preparing training data...")
        train_dataset, val_dataset = prepare_datasets()
        if train_dataset is None:
            raise Exception("Failed to prepare datasets")

        # Configure LoRA
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1, bias="none",
            task_type="CAUSAL_LM", inference_mode=False,
            fan_in_fan_out=True
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir="/mnt/storage/mistral_checkpoints",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            save_strategy="steps",
            save_steps=20000, save_total_limit=5,
            evaluation_strategy="steps",
            eval_steps=5000, learning_rate=2e-5,
            weight_decay=0.01, fp16=True,
            gradient_checkpointing=True,
            deepspeed="ds_config.json",
            dataloader_num_workers=4,
            report_to="tensorboard",
            logging_steps=100
        )

        # Save configuration
        output_dir = "lora_checkpoint"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(training_args, f"{output_dir}/training_args.pt")
        train_dataset.save_to_disk(f"{output_dir}/dataset")

        print("\n✅ LoRA preparation complete!")
        print(f"Dataset size: {len(train_dataset)} examples")
    except Exception as e:
        print(f"\n❌ Error preparing for LoRA: {e}")

if __name__ == "__main__":
    prepare_for_lora()
