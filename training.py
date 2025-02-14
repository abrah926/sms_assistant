from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import TrainingExample
import torch
from config import DEVICE, LLM_MODEL_PATH
import numpy as np
import os
import json
from datetime import datetime
import random
import asyncio
import shutil
import glob

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
    
    print("\n=== Training Data Quality Report ===")
    print(f"Total examples found: {len(examples)}")
    
    if examples:
        # Show 3 random examples
        sample = random.sample(examples, min(3, len(examples)))
        for i, ex in enumerate(sample, 1):
            print(f"\nExample {i}:")
            print(f"Customer: {ex.customer_message}")
            print(f"Agent: {ex.agent_response}")
            print(f"Metadata: {ex.meta_info}")
            print("-" * 50)
        
        # Analyze data quality
        print("\nData Analysis:")
        print(f"Average customer message length: {sum(len(ex.customer_message) for ex in examples)/len(examples):.1f} chars")
        print(f"Average agent response length: {sum(len(ex.agent_response) for ex in examples)/len(examples):.1f} chars")
        print(f"Examples with metadata: {sum(1 for ex in examples if ex.meta_info)}/{len(examples)}")
        
        # Check language distribution
        spanish = sum(1 for ex in examples if any(word in ex.agent_response.lower() for word in ['¿', 'é', 'ñ', 'ó']))
        print(f"Spanish responses: {spanish}/{len(examples)} ({spanish/len(examples)*100:.1f}%)")
    
    return [{
        "customer_message": ex.customer_message,
        "agent_response": ex.agent_response,
        "metadata": ex.meta_info
    } for ex in examples]

def log_message(msg: str, level: str = "INFO") -> None:
    """Unified logging function"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}][{level}] {msg}"
    print(formatted_msg)

class SalesModelTrainer:
    def __init__(self, model, tokenizer):
        if model is None or tokenizer is None:
            raise ValueError("Model and tokenizer must be provided")
            
        # Try loading from local first
        try:
            print("Loading model from local storage...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "models/mistral",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)
            self.tokenizer = AutoTokenizer.from_pretrained("models/mistral")
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("Falling back to original model")
            self.model = model.to(DEVICE)
            self.tokenizer = tokenizer

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Better learning rate management
        self.warmup_steps = 100
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)  # Start with lower learning rate
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-5,
            total_steps=self.warmup_steps,
            pct_start=0.3
        )
        
        # Initialize best responses with correct structure
        self.best_responses = {}  # Dict[str, List[Dict]]
        
        # Initialize checkpoint tracking
        self.latest_checkpoint = self._get_latest_checkpoint()
        if self.latest_checkpoint:
            self._load_checkpoint(self.latest_checkpoint)
        
        # Adjust training parameters
        self.min_iterations = 50
        self.max_iterations = 100
        self.convergence_threshold = 0.001
        self.improvement_window = 10
        self.target_score = 0.99
        self.checkpoint_frequency = 1  # Changed from 5 to 1 - save every iteration
        
        # Initialize training metrics
        self.evaluation_metrics = {
            'naturalness': 0.5,  # Start with baseline
            'sales_effectiveness': 0.5,
            'language_consistency': 0.5
        }
        
        # Add debug info
        print("\nModel Configuration:")
        print(f"Model name: {LLM_MODEL_PATH}")
        print(f"Tokenizer max length: {tokenizer.model_max_length}")
        print(f"Target score: {self.target_score}")
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        self.ab_test_results = []  # Store A/B test results
        
        # Expand Puerto Rican Spanish variations
        self.pr_spanish_responses = {
            "greetings": [
                "¡Wepa! ¿En qué te puedo ayudar?",
                "¡Saludos! ¿Qué metal estás buscando, pana?",
                "¡Bendiciones! ¿Qué necesitas hoy?",
                "¡Qué tal, boricua! ¿En qué te sirvo?",
                "¡Dimelo! ¿Qué estás buscando?",
                "¡Oye! ¿En qué te puedo ayudar?",
                "¿Qué hay, compai? ¿Qué necesitas?"
            ],
            "closings": [
                "¿Bregar con el pedido ahora?",
                "¿Te monto los números?",
                "¿Hacemos el deal?",
                "¿Te preparo el estimate?",
                "¿Lo hacemos?",
                "¿Qué me dices, lo bregamos?",
                "¿Te tiro el breakdown de los precios?"
            ],
            "casual_responses": [
                "Mira, ese precio está brutal",
                "Te puedo dar un deal bien bueno",
                "Ese metal está volando, pana",
                "Tremenda calidad, de verdad",
                "Está cabrón ese material",
                "El precio está al bate",
                "Eso está de show"
            ],
            "spanglish": [
                "Te puedo hacer un mejor deal si ordenas más quantity",
                "Ese price está hot. Podemos shippear rápido también",
                "Si hacemos el order hoy, te doy un discount brutal",
                "El delivery es gratis si ordenas over 100 tons",
                "Tenemos special pricing pa' bulk orders",
                "Te puedo hacer layaway si prefieres",
                "El quality check viene incluido"
            ],
            "negotiations": [
                "Dale, podemos trabajar con eso",
                "Mira, si coges más te puedo dar mejor precio",
                "Vamos a bregarlo, ¿qué tal si...?",
                "Te puedo hacer un descuentito ahí",
                "Podemos llegar a un middle ground"
            ],
            "follow_ups": [
                "¿Qué pasó con el estimate que te envié?",
                "¿Pudiste chequear los números?",
                "¿Cómo vamos con eso, pana?",
                "¿Ya decidiste qué vas a hacer?"
            ]
        }

        # Add log counter
        self.log_counter = 0

    def _get_latest_checkpoint(self):
        """Get the most recent checkpoint file"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                return None
            
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith("checkpoint_") and f.endswith(".pt")]
            if not checkpoints:
                return None
            
            latest = max(checkpoints, 
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
            return os.path.join(self.checkpoint_dir, latest)
        except Exception as e:
            print(f"Error finding latest checkpoint: {e}")
            return None

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

    def backup_checkpoint(self, checkpoint_path: str):
        """Create a backup of the checkpoint"""
        try:
            # Create backups directory if it doesn't exist
            backup_dir = "checkpoints/backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{backup_dir}/backup_{timestamp}.pt"
            
            # Copy checkpoint to backup
            shutil.copy2(checkpoint_path, backup_path)
            print(f"Created backup: {backup_path}")
            
            # Keep only last 5 backups to save space
            backups = sorted(glob.glob(f"{backup_dir}/backup_*.pt"))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    os.remove(old_backup)
                
        except Exception as e:
            print(f"Error creating backup: {e}")

    async def save_checkpoint(self, iteration: int, scores: list, examples: list):
        """Save training checkpoint with verification"""
        try:
            checkpoint = {
                'iteration': iteration,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'scores': scores,
                'best_responses': self.best_responses,
                'timestamp': datetime.now().isoformat(),
                'examples_processed': len(examples),
                'running_accuracy': sum(scores) / len(scores) if scores else 0
            }
            
            # Save main checkpoint
            path = f"{self.checkpoint_dir}/checkpoint_{iteration}.pt"
            torch.save(checkpoint, path)
            
            # Verify save
            if not os.path.exists(path):
                raise Exception(f"Checkpoint file not created: {path}")
            
            # Verify can load
            test_load = torch.load(path)
            if 'iteration' not in test_load:
                raise Exception("Checkpoint data incomplete")
            
            print(f"✓ Checkpoint saved and verified: {path}")
            
            # Create backup with verification
            backup_dir = os.path.join(self.checkpoint_dir, "backups")
            backup_path = f"{backup_dir}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            shutil.copy2(path, backup_path)
            
            if not os.path.exists(backup_path):
                raise Exception(f"Backup not created: {backup_path}")
            
            print(f"✓ Backup created and verified: {backup_path}")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            raise  # Make sure we see the error

    def _load_checkpoint(self, checkpoint_path):
        """Load a checkpoint file"""
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load model and training states
            self.model.load_state_dict(checkpoint['model_state'])
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            self.best_responses = checkpoint.get('best_responses', {})
            
            print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
            print(f"Previous best responses: {len(self.best_responses)}")
            
            return checkpoint['iteration']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with fresh model state")
            return 0

    async def train_iteratively(self, session: AsyncSession, iterations: int = 50, status_callback = None):
        try:
            # Initial setup logs
            print("\n=== Training Pipeline Check ===")
            print(f"[1] Model: {LLM_MODEL_PATH}")
            print(f"[2] Device: {DEVICE}")
            print(f"[3] Model loaded: {self.model is not None}")
            print(f"[4] Tokenizer loaded: {self.tokenizer is not None}")
            
            # Test generation
            print("\n[5] Testing model generation...")
            test_input = "Hello, I need steel plates"
            try:
                test_response = await self.generate_response(test_input)
                print(f"Test output: {test_response[:100]}...")
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                return None

            # Load training data
            examples = await prepare_training_data(session)
            if not examples:
                print("❌ [6] No training examples found!")
                return None
            
            print(f"\n[7] Training data loaded: {len(examples)} examples")
            
            # Track metrics
            best_accuracy = 0.0
            running_accuracy = 0.0
            validation_losses = []
            
            for iteration in range(1, iterations + 1):
                print(f"\n[{iteration+7}] === Iteration {iteration}/{iterations} ===")
                
                # Training loop
                iteration_scores = []
                for idx, ex in enumerate(examples, 1):
                    response = await self.generate_response(ex['customer_message'])
                    scores = self.score_response(response, ex['customer_message'])
                    score = sum(scores.values()) / len(scores)
                    iteration_scores.append(score)
                    
                    # Learn from examples silently
                    if score < 0.8:  # Only show scores below threshold
                        print(f"Sample {idx}: Score {score*100:.1f}%")
                
                # Show iteration metrics
                current_accuracy = sum(iteration_scores) / len(iteration_scores)
                running_accuracy = (running_accuracy * (iteration - 1) + current_accuracy) / iteration
                best_accuracy = max(best_accuracy, running_accuracy)
                
                print(f"\nMetrics:")
                print(f"- Current: {current_accuracy*100:.1f}%")
                print(f"- Running: {running_accuracy*100:.1f}%")
                print(f"- Best: {best_accuracy*100:.1f}%")
                
                # Validation
                validation_loss = await self.validate_model(examples)
                print(f"- Validation loss: {validation_loss:.4f}")
                
                # Early stopping check
                validation_losses.append(validation_loss)
                if len(validation_losses) > 3 and all(validation_losses[-3:] > validation_losses[-4:-1]):
                    print("\n⚠️ Stopping early - validation loss increasing")
                    break
                
                await asyncio.sleep(0.5)
            
            print("\n=== Training Complete ===")
            print(f"Final accuracy: {running_accuracy*100:.1f}%")
            print(f"Best accuracy: {best_accuracy*100:.1f}%")
            return self.model
            
        except Exception as e:
            print(f"\n❌ Error in training: {str(e)}")
            raise

    def _should_stop_training(self, iteration: int, scores: list, stagnant_iterations: int) -> bool:
        """Determine if training should stop"""
        # Must complete minimum iterations
        if iteration < self.min_iterations:
            return False
            
        # Check if reached max iterations
        if iteration >= self.max_iterations:
            print("Reached maximum iterations")
            return True
            
        # Check if reached very high target score
        if scores[-1] >= self.target_score:
            print(f"Reached target score of {self.target_score}!")
            return True
            
        # Check for extended stagnation
        if stagnant_iterations >= self.improvement_window:
            print(f"No improvements for {self.improvement_window} iterations")
            return True
            
        # Check if improvements are minimal over window
        if len(scores) >= self.improvement_window:
            recent_improvements = [
                scores[i] - scores[i-1] 
                for i in range(-1, -self.improvement_window, -1)
            ]
            if all(imp < self.convergence_threshold for imp in recent_improvements):
                print(f"Improvements below threshold of {self.convergence_threshold}")
                return True
                
        return False

    async def generate_and_improve(self, examples):
        """Generate responses and improve them based on metrics"""
        improved = []
        
        for ex in examples:
            # Generate response
            response = await self.generate_response(ex['customer_message'])
            
            # Score response
            scores = self.score_response(response, ex['customer_message'])
            
            # Improve if needed
            if not self.is_human_like(scores):
                response = await self.improve_response(response, scores)
            
            improved.append({
                'customer_message': ex['customer_message'],
                'agent_response': response,
                'scores': scores
            })
            
        return improved

    def score_response(self, response: str, customer_message: str) -> Dict[str, float]:
        """Score response quality with language checks"""
        try:
            # Reject responses with non-English/Spanish characters
            if any(ord(c) > 127 and c not in 'áéíóúüñ¿¡' for c in response):
                return {'naturalness': 0, 'sales_effectiveness': 0, 'language_consistency': 0}
            
            scores = {}
            
            # Calculate scores silently
            natural_score = self.score_naturalness(response)
            sales_score = self.score_sales_effectiveness(response)
            lang_score = self.score_language_consistency(response)
            
            scores['naturalness'] = natural_score * 0.3
            scores['sales_effectiveness'] = sales_score * 0.4
            scores['language_consistency'] = lang_score * 0.3
            
            total_score = sum(scores.values())
            
            # Log to file if score is good
            if total_score > 0.8:
                self.log_best_response(customer_message, response, total_score)
            
            return scores
            
        except Exception as e:
            print(f"Error in scoring: {e}")
            return {'error': 0.0}

    def score_naturalness(self, response: str) -> float:
        """Score how natural the response sounds"""
        score = 0.0
        
        # Check for conversational elements
        if any(greeting in response.lower() for greeting in ["hello", "hi", "hey", "hola", "saludos"]):
            score += 0.2
        
        # Check for proper punctuation and structure
        if "?" in response and "!" in response:  # Engaging punctuation
            score += 0.2
        
        # Check for natural flow markers
        flow_markers = ["well", "so", "now", "also", "but", "and", "or", "pues", "entonces"]
        if any(marker in response.lower() for marker in flow_markers):
            score += 0.2
        
        # Check response length (too short or too long is unnatural)
        words = len(response.split())
        if 10 <= words <= 50:  # Good length for a response
            score += 0.2
        
        # Check for personalization
        if any(personal in response.lower() for personal in ["you", "your", "tu", "usted", "su"]):
            score += 0.2
        
        print(f"Naturalness breakdown: {score*100:.2f}%")
        return min(score, 1.0)

    def score_sales_effectiveness(self, response: str) -> float:
        """Score sales effectiveness of response"""
        score = 0.0
        
        # Check for price/deal discussion
        if any(term in response.lower() for term in ["price", "deal", "discount", "offer", "special", "precio", "descuento"]):
            score += 0.25
        
        # Check for call to action
        if any(term in response.lower() for term in ["order", "buy", "purchase", "get", "comprar", "ordenar"]):
            score += 0.25
        
        # Check for product information
        if any(term in response.lower() for term in ["quality", "features", "benefits", "calidad", "beneficios"]):
            score += 0.25
        
        # Check for urgency/closing
        if any(term in response.lower() for term in ["today", "now", "limited", "ahora", "hoy", "pronto"]):
            score += 0.25
        
        print(f"Sales effectiveness breakdown: {score*100:.2f}%")
        return min(score, 1.0)

    def score_language_consistency(self, response: str) -> float:
        """Score language consistency (Puerto Rican Spanish/Spanglish)"""
        score = 0.0
        
        # Check for Puerto Rican expressions
        pr_expressions = sum(1 for exp in self.pr_spanish_responses["casual_responses"] 
                           if exp.lower() in response.lower())
        score += min(pr_expressions * 0.2, 0.4)
        
        # Check for Spanglish terms
        spanglish_terms = sum(1 for term in self.pr_spanish_responses["spanglish"] 
                            if term.lower() in response.lower())
        score += min(spanglish_terms * 0.2, 0.3)
        
        # Check for Spanish grammar structure
        if any(term in response.lower() for term in ["que", "como", "donde", "cuando"]):
            score += 0.3
            
        return min(score, 1.0)

    def score_cultural_fit(self, response: str) -> float:
        """Score cultural appropriateness for Puerto Rican market"""
        score = 0.0
        
        # Check for cultural markers
        cultural_terms = ["pana", "boricua", "wepa", "bendiciones", "brutal"]
        matches = sum(1 for term in cultural_terms if term in response.lower())
        score += min(matches * 0.2, 0.4)
        
        # Check for industry-specific Puerto Rican terms
        industry_terms = ["bildin", "site", "delivery", "order", "quote"]
        matches = sum(1 for term in industry_terms if term in response.lower())
        score += min(matches * 0.2, 0.3)
        
        # Check for appropriate formality level
        if not any(formal in response.lower() for formal in ["usted", "cordialmente", "estimado"]):
            score += 0.3  # Puerto Rican business Spanish tends to be less formal
            
        return min(score, 1.0)

    def show_best_responses(self):
        """Display current best responses"""
        print("\n=== Current Best Responses ===")
        if not self.best_responses:
            print("No best responses stored yet")
            return
        
        print(f"Total best responses stored: {len(self.best_responses)}")
        # Show 3 random examples
        sample_keys = random.sample(list(self.best_responses.keys()), min(3, len(self.best_responses)))
        for i, key in enumerate(sample_keys, 1):
            print(f"\nBest Response {i}:")
            print(f"Customer: {key}")
            print(f"Agent: {self.best_responses[key]}")
            print("-" * 50)

    async def learn_from_example(self, customer_message: str, good_response: str):
        """Learn from a good example with gradient accumulation"""
        try:
            self.model.train()
            accumulated_loss = 0
            accumulation_steps = 4  # Accumulate over 4 steps for stability
            
            for _ in range(accumulation_steps):
                # Format with slight variations for robustness
                conversation = f"""System: You are a professional metal sales agent.
Customer: {customer_message}
Agent: {good_response}"""

                inputs = self.tokenizer(
                    conversation,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(DEVICE)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / accumulation_steps  # Scale loss
                accumulated_loss += loss.item()
                
                # Backward pass
                loss.backward()
            
            # Clip gradients and optimize
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            print(f"Learned from example (avg loss: {accumulated_loss/accumulation_steps:.4f})")
            return accumulated_loss/accumulation_steps
            
        except Exception as e:
            print(f"Error learning from example: {e}")
            return None

    async def generate_response(self, message: str) -> str:
        try:
            # Mistral v0.2 specific prompt format
            messages = [
                {"role": "system", "content": "You are a professional metal sales agent. You are knowledgeable about steel, aluminum, copper and other metals. Always be professional and focus on making sales."},
                {"role": "user", "content": message}
            ]
            
            # Use Mistral's chat template with attention mask
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            )
            
            attention_mask = torch.ones_like(model_inputs)  # Create attention mask
            inputs = {
                "input_ids": model_inputs.to(DEVICE),
                "attention_mask": attention_mask.to(DEVICE)
            }
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Clean up response to get only the assistant's part
            response = response.split("[/INST]")[-1].strip()
            return response
            
        except Exception as e:
            print(f"Error generating: {e}")
            return ""

    def is_human_like(self, scores: Dict[str, float], threshold: float = 0.8) -> bool:
        """Check if response meets human-like threshold"""
        total_score = sum(scores.values()) / len(scores)
        return total_score >= threshold

    async def improve_response(self, response: str, scores: Dict[str, float]) -> str:
        """Improve a response based on scores"""
        try:
            # If response needs improvement, generate a new one with more context
            prompt = f"""Improve this response to be more natural and effective:
Previous response: {response}
Make it more: {', '.join(k for k, v in scores.items() if v < 0.8)}
"""
            return await self.generate_response(prompt)
        except Exception as e:
            log_message(f"Error improving response: {e}", "ERROR")
            return response

    async def validate_model(self, validation_examples):
        """Run validation on a subset of examples"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for ex in validation_examples:
                response = await self.generate_response(ex['customer_message'])
                scores = self.score_response(response, ex['customer_message'])
                total_loss += 1 - (sum(scores.values()) / len(scores))
        return total_loss / len(validation_examples)

    def log_best_response(self, customer_message: str, response: str, score: float):
        """Log best responses to file"""
        try:
            log_path = "logs/best_responses.jsonl"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "customer": customer_message,
                "response": response,
                "score": score
            }
            
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
        except Exception as e:
            print(f"Error logging best response: {e}")