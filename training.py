from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from typing import List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import TrainingExample
import torch
from config import DEVICE, LLM_MODEL_PATH
import numpy as np

class SalesConversationDataset(Dataset):
    def __init__(self, tokenizer, examples: List[Dict], max_length: int = 512):
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        
        # Format conversation
        conversation = f"""System: You are a professional metal sales agent.
Customer: {example['customer_message']}
Agent: {example['agent_response']}"""

        # Tokenize
        encodings = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

async def prepare_training_data(session: AsyncSession) -> List[Dict]:
    """Get training examples from database"""
    query = select(TrainingExample).order_by(TrainingExample.created_at.desc())
    result = await session.execute(query)
    examples = result.scalars().all()
    
    return [{
        "customer_message": ex.customer_message,
        "agent_response": ex.agent_response,
        "metadata": ex.metadata
    } for ex in examples]

class SalesModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    async def train(self, session: AsyncSession, output_dir: str = "trained_model"):
        """Train the model on collected examples"""
        print("Preparing training data...")
        examples = await prepare_training_data(session)
        
        if not examples:
            print("No training examples found!")
            return
        
        print(f"Found {len(examples)} training examples")
        
        # Create dataset
        dataset = SalesConversationDataset(
            self.tokenizer,
            examples
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=dataset
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

    def evaluate_response(self, response: str, expected: str) -> float:
        """Evaluate response quality"""
        # Simple overlap score for now
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        
        overlap = len(response_words.intersection(expected_words))
        total = len(response_words.union(expected_words))
        
        return overlap / total if total > 0 else 0 