import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, concatenate_datasets
import os
import json
from models import TrainingExample
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

def prepare_datasets():
    """Prepare datasets with multi-turn context and sales flow"""
    print("\n=== Preparing Datasets ===")
    
    try:
        # Load our existing training data
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get examples from our database
        examples = session.query(TrainingExample).all()
        print(f"Loaded {len(examples)} examples from database")
        
        # Group conversations by customer_id to maintain context
        conversations = {}
        for ex in examples:
            if ex.customer_id not in conversations:
                conversations[ex.customer_id] = []
            conversations[ex.customer_id].append({
                "customer": ex.customer_message,
                "agent": ex.agent_response,
                "meta": ex.meta_info
            })
        
        # Format with multi-turn context
        all_examples = []
        system_prompt = """You are an elite VIP CEO-level sales representative.
Follow this sales process:
1. Engage naturally and build rapport
2. Identify customer needs through subtle questions
3. Present solutions with social proof
4. Handle objections using urgency and exclusivity
5. Guide smoothly to checkout
6. Follow up for retention

Remember past context and use customer data to personalize responses."""

        for customer_id, msgs in conversations.items():
            # Create sliding window of conversations (3 turns at a time)
            for i in range(len(msgs)):
                context = msgs[max(0, i-2):i+1]  # Get up to 2 previous messages
                
                history = []
                for msg in context[:-1]:  # Add conversation history
                    history.append(f"Customer: {msg['customer']}")
                    history.append(f"CEO Salesman: {msg['agent']}")
                
                current = context[-1]
                
                # Format with full context
                input_text = f"{system_prompt}\n\n"
                if history:
                    input_text += "Previous conversation:\n"
                    input_text += "\n".join(history) + "\n\n"
                input_text += f"Customer: {current['customer']}"
                
                # Add any CRM/meta info if available
                if current['meta']:
                    input_text += f"\n[CRM Data: {current['meta']}]"
                
                example = {
                    "input": input_text,
                    "output": f"CEO Salesman: {current['agent']}"
                }
                all_examples.append(example)
        
        # Clean and split data
        print(f"\nCreated {len(all_examples)} training examples with context")
        dataset = Dataset.from_list(all_examples)
        splits = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        
        return splits["train"], splits["test"]
        
    except Exception as e:
        print(f"Error preparing datasets: {e}")
        return None, None

def prepare_for_lora():
    print("\n=== Preparing for LoRA Training ===")
    
    try:
        # 1. Load base model with 8-bit quantization
        print("1. Loading Mistral-7B with quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "models/mistral",
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("models/mistral")
        
        # 2. Prepare datasets
        print("\n2. Preparing training data...")
        train_dataset, val_dataset = prepare_datasets()
        if train_dataset is None:
            raise Exception("Failed to prepare datasets")
        
        # 3. Configure LoRA for multi-GPU
        print("\n3. Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 4. Configure training arguments for A100s
        training_args = TrainingArguments(
            output_dir="./mistral-finetuned-lora",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=10,
            save_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=True,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            deepspeed="ds_config.json"
        )
        
        # 5. Save everything
        output_dir = "lora_checkpoint"
        os.makedirs(output_dir, exist_ok=True)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(training_args, f"{output_dir}/training_args.pt")
        
        # Save dataset
        train_dataset.save_to_disk(f"{output_dir}/dataset")
        
        # 6. Create DeepSpeed config
        ds_config = {
            "fp16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"}
            },
            "gradient_accumulation_steps": 2,
        }
        
        with open("ds_config.json", "w") as f:
            json.dump(ds_config, f)
        
        print(f"\n✅ LoRA preparation complete! Saved to {output_dir}/")
        print(f"Dataset size: {len(train_dataset)} examples")
        print("\nOptimized for:")
        print("- 10x A100 GPUs")
        print("- 8-bit quantization")
        print("- DeepSpeed ZeRO-3")
        print("- Gradient checkpointing")
        
    except Exception as e:
        print(f"\n❌ Error preparing for LoRA: {e}")

if __name__ == "__main__":
    prepare_for_lora() 