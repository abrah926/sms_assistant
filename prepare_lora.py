import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, concatenate_datasets
import os
import json
from models import TrainingExample, TrainingCheckpoint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from datetime import datetime
from pathlib import Path

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

class CheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, best_responses_file="logs/best_responses.jsonl"):
        self.tokenizer = tokenizer
        self.best_responses_file = best_responses_file
        Path("logs").mkdir(exist_ok=True)
        self.best_score = 0.0
        
    def on_save(self, args, state, control, model, **kwargs):
        print(f"\n=== Checkpoint {state.global_step} Analysis ===")
        
        try:
            # Test prompts with different scenarios
            test_prompts = [
                # Price inquiries
                "What's your best price on 500kg of steel?",
                "Can you beat $5/kg for copper wire?",
                
                # Urgency scenarios
                "I need this delivered by next week",
                "My current supplier is out of stock",
                
                # Objection handling
                "That's a bit expensive",
                "I'll think about it and get back to you",
                
                # Technical questions
                "What's the tensile strength of your steel?",
                "Do you have certification for aerospace grade?",
                
                # Follow-up scenarios
                "I ordered last month, need something similar",
                "Still waiting for that quote you promised"
            ]
            
            scores = []
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
            
            for prompt in test_prompts:
                # Generate with system prompt
                full_prompt = f"""You are an elite VIP CEO-level sales representative.
                Customer: {prompt}
                CEO:"""
                
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Detailed scoring
                score_details = self.score_response(response, prompt)
                scores.append(score_details["total"])
                
                # Save if it's a good response
                if score_details["total"] > 0.8:
                    self.save_best_response(prompt, response, score_details)
            
            avg_score = sum(scores) / len(scores)
            
            # Log detailed metrics
            print(f"\n📊 Checkpoint Performance:")
            print(f"Step: {state.global_step:,}")
            print(f"Average Score: {avg_score*100:.1f}%")
            print(f"Training Loss: {state.log_history[-1].get('loss', 'N/A'):.4f}")
            print(f"Learning Rate: {state.log_history[-1].get('learning_rate', 'N/A'):.2e}")
            print(f"GPU Memory: {gpu_mem:.1f}GB")
            
            # Save if best checkpoint
            if avg_score > self.best_score:
                self.best_score = avg_score
                print(f"✨ New best score! Saving checkpoint...")
                model.save_pretrained(f"checkpoints/best_{state.global_step}")
            
        except Exception as e:
            print(f"Error in checkpoint analysis: {e}")
    
    def score_response(self, response: str, prompt: str) -> dict:
        """Detailed response scoring"""
        scores = {
            "sales_elements": 0.0,
            "personalization": 0.0,
            "professionalism": 0.0,
            "call_to_action": 0.0,
            "context_awareness": 0.0
        }
        
        # Sales elements (30%)
        sales_words = ["price", "offer", "deal", "quote", "discount", "value", "investment"]
        scores["sales_elements"] = min(sum(word in response.lower() for word in sales_words) * 0.1, 0.3)
        
        # Personalization (20%)
        personal_elements = ["you", "your", "specifically", "custom", "needs", "requirements"]
        scores["personalization"] = min(sum(word in response.lower() for word in personal_elements) * 0.05, 0.2)
        
        # Professionalism (20%)
        unprofessional = ["um", "uh", "like", "sort of", "kind of", "maybe"]
        scores["professionalism"] = 0.2 if not any(word in response.lower() for word in unprofessional) else 0.0
        
        # Call to action (15%)
        cta_words = ["order", "purchase", "buy", "secure", "reserve", "confirm", "proceed"]
        scores["call_to_action"] = min(sum(word in response.lower() for word in cta_words) * 0.05, 0.15)
        
        # Context awareness (15%)
        context_words = [word for word in prompt.lower().split() if len(word) > 4]
        scores["context_awareness"] = min(sum(word in response.lower() for word in context_words) * 0.03, 0.15)
        
        # Calculate total
        scores["total"] = sum(scores.values())
        return scores
    
    def save_best_response(self, prompt: str, response: str, scores: dict):
        """Save high-quality responses with detailed metrics"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "scores": scores,
            "checkpoint_step": state.global_step,
            "training_loss": state.log_history[-1].get('loss', 'N/A')
        }
        
        with open(self.best_responses_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

def prepare_for_lora():
    print("\n=== Preparing for LoRA Training on 5x A100s ===")
    
    try:
        # 1. Load base model with A100 optimizations
        print("1. Loading Mistral-7B...")
        model = AutoModelForCausalLM.from_pretrained(
            "models/mistral",
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            use_flash_attention_2=True,  # A100 optimization
            max_memory={0: "70GB"}  # Reserve memory for 80GB A100
        )
        tokenizer = AutoTokenizer.from_pretrained("models/mistral")
        
        # 2. Prepare datasets
        print("\n2. Preparing training data...")
        train_dataset, val_dataset = prepare_datasets()
        if train_dataset is None:
            raise Exception("Failed to prepare datasets")
        
        # 3. Configure LoRA for optimal A100 usage
        lora_config = LoraConfig(
            r=16,  # Increased for A100
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
            fan_in_fan_out=True  # Better initialization for A100
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 4. Configure training arguments for checkpoint persistence
        training_args = TrainingArguments(
            output_dir="/mnt/storage/mistral_checkpoints",  # Persistent storage
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            save_strategy="steps",
            save_steps=20000,  # Save every 20K examples
            save_total_limit=5,  # Keep last 5 checkpoints
            evaluation_strategy="steps",
            eval_steps=5000,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=True,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            deepspeed="ds_config.json",
            dataloader_num_workers=4,
            report_to="tensorboard",
            logging_steps=100,
            # Add checkpoint tracking in database
            hub_strategy="every_save",
            push_to_hub=False  # We'll handle saving manually
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
            "fp16": {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "overlap_comm": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True
            },
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
            "train_batch_size": 16 * 5,  # batch_size * num_gpus
            "train_micro_batch_size_per_gpu": 16
        }
        
        with open("ds_config.json", "w") as f:
            json.dump(ds_config, f)
        
        print(f"\n✅ LoRA preparation complete! Saved to {output_dir}/")
        print(f"Dataset size: {len(train_dataset)} examples")
        print("\nOptimized for:")
        print("- 5x A100 GPUs")
        print("- 8-bit quantization")
        print("- DeepSpeed ZeRO-3")
        print("- Gradient checkpointing")
        
        # Add checkpoint tracking to database
        def save_checkpoint_to_db(state, control, model, **kwargs):
            try:
                checkpoint_path = f"/mnt/storage/mistral_checkpoints/checkpoint-{state.global_step}"
                model.save_pretrained(checkpoint_path)
                
                # Save checkpoint info to database
                with Session() as session:
                    checkpoint = TrainingCheckpoint(
                        step=state.global_step,
                        path=checkpoint_path,
                        loss=state.log_history[-1].get('loss', 0),
                        created_at=datetime.now()
                    )
                    session.add(checkpoint)
                    session.commit()
                    
                print(f"✓ Saved checkpoint at step {state.global_step}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        
        # Add callback for checkpoint saving
        class CheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                save_checkpoint_to_db(state, control, kwargs.get('model'))
                return control
        
        callbacks = [CheckpointCallback()]
        
        # Create trainer with callbacks
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks
        )
        
    except Exception as e:
        print(f"\n❌ Error preparing for LoRA: {e}")

if __name__ == "__main__":
    prepare_for_lora() 