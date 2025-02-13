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
import os
import json
from datetime import datetime
import random
import asyncio

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
    
    # Add debug logging
    print(f"\nChecking training data quality...")
    print(f"Total examples: {len(examples)}")
    if examples:
        print("Sample example:")
        print(f"Customer: {examples[0].customer_message}")
        print(f"Agent: {examples[0].agent_response}")
        print(f"Metadata: {examples[0].meta_info}")
    
    return [{
        "customer_message": ex.customer_message,
        "agent_response": ex.agent_response,
        "metadata": ex.meta_info
    } for ex in examples]

class SalesModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize checkpoint tracking
        self.latest_checkpoint = self._get_latest_checkpoint()
        if self.latest_checkpoint:
            self._load_checkpoint(self.latest_checkpoint)
        
        # Adjust training parameters for DeepSeek
        self.min_iterations = 50  # Reduced from 1000
        self.max_iterations = 100  # Reduced from 2000
        self.convergence_threshold = 0.001  # Less strict
        self.improvement_window = 10  # Shorter window
        self.target_score = 0.99
        self.checkpoint_frequency = 5  # Save more frequently
        
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
        self.best_responses = {}  # Store best responses for A/B comparison
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

    async def save_checkpoint(self, iteration: int, scores: list, examples: list):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state': self.model.state_dict(),
            'scores': scores,
            'best_responses': self.best_responses,
            'timestamp': datetime.now().isoformat(),
            'examples_processed': len(examples)
        }
        
        path = f"{self.checkpoint_dir}/checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load a checkpoint file"""
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.ab_test_results = checkpoint.get('ab_test_results', [])
            self.best_responses = checkpoint.get('best_responses', {})
            
            # Add more detailed logging
            print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
            print(f"Previous best responses: {len(self.best_responses)}")
            print(f"Previous AB test results: {len(self.ab_test_results)}")
            
            return checkpoint['iteration']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with fresh model state")
            return 0

    async def train_iteratively(self, session: AsyncSession, iterations: int = 50, status_callback = None):
        try:
            log_file = "logs/training.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            def log_message(msg, level="INFO"):
                self.log_counter += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_msg = f"[{self.log_counter:04d}][{timestamp}][{level}] {msg}"
                
                with open(log_file, "a") as f:
                    f.write(formatted_msg + "\n")
                print(formatted_msg)
            
            log_message("=== Starting Training Session ===", "START")
            
            # Check for existing checkpoints
            if self.latest_checkpoint:
                log_message(f"Found existing checkpoint: {self.latest_checkpoint}")
                start_iteration = self._load_checkpoint(self.latest_checkpoint)
                log_message(f"Resuming from iteration {start_iteration}")
            else:
                log_message("No existing checkpoint found - starting fresh", "WARN")
                start_iteration = 0
            
            # Get training examples
            examples = await prepare_training_data(session)
            if not examples:
                log_message("No training examples found!", "ERROR")
                return None
            
            log_message(f"Training with {len(examples)} examples")
            
            # Track best accuracy
            best_accuracy = 0.0
            running_accuracy = 0.0
            
            for iteration in range(start_iteration + 1, iterations + 1):
                log_message(f"Starting iteration {iteration}/{iterations}", "ITER")
                
                # Training logic
                sample_size = min(100, len(examples))
                sample = random.sample(examples, sample_size)
                
                # Process samples with progress
                iteration_scores = []
                for idx, ex in enumerate(sample, 1):
                    if idx % 10 == 0:  # Log every 10 samples
                        log_message(f"Processing sample {idx}/{sample_size}")
                    
                    response = await self.generate_response(ex['customer_message'])
                    scores = self.score_response(response)
                    iteration_scores.append(sum(scores.values()) / len(scores))
                
                current_accuracy = sum(iteration_scores) / len(iteration_scores)
                running_accuracy = (running_accuracy * (iteration - 1) + current_accuracy) / iteration
                best_accuracy = max(best_accuracy, running_accuracy)
                
                log_message(f"Iteration {iteration} Results:", "METRICS")
                log_message(f"  Current Accuracy: {current_accuracy*100:.2f}%")
                log_message(f"  Running Average: {running_accuracy*100:.2f}%")
                log_message(f"  Best Accuracy: {best_accuracy*100:.2f}%")
                
                if status_callback:
                    status_callback(iteration, running_accuracy)
                
                if iteration % self.checkpoint_frequency == 0:
                    log_message(f"Saving checkpoint at iteration {iteration}", "SAVE")
                    await self.save_checkpoint(iteration, iteration_scores, examples)
                
                await asyncio.sleep(0.5)  # Prevent CPU overload
            
            log_message("=== Training Complete ===", "END")
            log_message(f"Final Accuracy: {running_accuracy*100:.2f}%")
            log_message(f"Best Accuracy: {best_accuracy*100:.2f}%")
            return self.model
            
        except Exception as e:
            log_message(f"Error in training: {str(e)}", "ERROR")
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
            scores = self.score_response(response)
            
            # Improve if needed
            if not self.is_human_like(scores):
                response = await self.improve_response(response, scores)
            
            improved.append({
                'customer_message': ex['customer_message'],
                'agent_response': response,
                'scores': scores
            })
            
        return improved

    def score_response(self, response: str) -> Dict[str, float]:
        """Score response quality with adjusted weights"""
        scores = {}
        
        # Naturalness (40% weight)
        scores['naturalness'] = self.score_naturalness(response) * 0.4
        
        # Sales effectiveness (40% weight)
        scores['sales_effectiveness'] = self.score_sales_effectiveness(response) * 0.4
        
        # Language consistency (20% weight)
        scores['language_consistency'] = self.score_language_consistency(response) * 0.2
        
        # Debug scoring
        print(f"\nScoring Response: {response[:100]}...")
        print(f"Individual scores: {scores}")
        print(f"Total score: {sum(scores.values())}")
        
        return scores

    def is_human_like(self, scores, threshold=0.9):
        """Check if response scores indicate human-like quality"""
        return all(score > threshold for score in scores.values())

    async def improve_response(self, response, scores):
        """Improve response based on scores"""
        improvements = []
        
        if scores['naturalness'] < 0.9:
            response = self.add_conversational_elements(response)
            
        if scores['sales_effectiveness'] < 0.9:
            response = self.enhance_sales_elements(response)
            
        if scores['language_consistency'] < 0.9:
            response = self.fix_language_consistency(response)
            
        if scores['cultural_accuracy'] < 0.9:
            response = self.adjust_cultural_elements(response)
            
        return response

    def check_convergence(self, old_metrics, new_metrics, threshold=0.001):
        """Check if model has converged to optimal performance"""
        improvements = [
            new_metrics[key] - old_metrics[key]
            for key in old_metrics
        ]
        return all(imp < threshold for imp in improvements)

    async def generate_ab_pairs(self, examples):
        """Generate A/B test pairs for each example"""
        ab_pairs = []
        
        for ex in examples:
            # Generate two different responses
            response_a = await self.generate_response(ex['customer_message'], style='standard')
            response_b = await self.generate_response(ex['customer_message'], style='enhanced')
            
            # Score both responses
            score_a = self.score_response(response_a)
            score_b = self.score_response(response_b)
            
            pair = {
                'customer_message': ex['customer_message'],
                'variation_a': {
                    'response': response_a,
                    'scores': score_a
                },
                'variation_b': {
                    'response': response_b,
                    'scores': score_b
                },
                'metadata': ex.get('metadata', {})
            }
            
            ab_pairs.append(pair)
            
            # Store the better response
            if sum(score_b.values()) > sum(score_a.values()):
                self.best_responses[ex['customer_message']] = response_b
            else:
                self.best_responses[ex['customer_message']] = response_a
                
        return ab_pairs

    async def save_ab_results(self):
        """Save A/B test results for review"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_examples': len(self.ab_test_results),
            'ab_pairs': self.ab_test_results
        }
        
        with open('ab_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\nA/B test results saved. You can review them in ab_test_results.json")

    async def generate_response(self, message: str, style: str = 'standard') -> str:
        """Generate a response using the model"""
        try:
            # Format input
            prompt = f"""System: You are a professional metal sales agent.
Customer: {message}
Agent:"""
            
            # Add style-specific instructions
            if style == 'enhanced':
                prompt = f"""System: You are a professional metal sales agent. Use Puerto Rican Spanish and Spanglish naturally.
Include common Puerto Rican expressions and industry terms.
Customer: {message}
Agent:"""
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = response.replace(prompt, "").strip()
            
            # Add Puerto Rican elements if enhanced style
            if style == 'enhanced':
                response = self.add_puerto_rican_elements(response)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Lo siento, hubo un error. ¿Puedes repetir la pregunta?"

    def add_puerto_rican_elements(self, response: str) -> str:
        """Add Puerto Rican expressions and style to response"""
        # Add greeting if not present
        if not any(g in response for g in self.pr_spanish_responses["greetings"]):
            response = f"{random.choice(self.pr_spanish_responses['greetings'])} {response}"
        
        # Add casual expression
        if random.random() > 0.5:
            response += f" {random.choice(self.pr_spanish_responses['casual_responses'])}"
        
        # Add closing if not present
        if not any(c in response for c in self.pr_spanish_responses["closings"]):
            response += f" {random.choice(self.pr_spanish_responses['closings'])}"
        
        return response

    def score_naturalness(self, response: str) -> float:
        """Score how natural the response sounds"""
        score = 0.0
        
        # Check for presence of greetings
        if any(g in response.lower() for g in self.pr_spanish_responses["greetings"]):
            score += 0.2
            
        # Check for casual expressions
        if any(c in response.lower() for c in self.pr_spanish_responses["casual_responses"]):
            score += 0.2
            
        # Check for natural closings
        if any(c in response.lower() for c in self.pr_spanish_responses["closings"]):
            score += 0.2
            
        # Check for Spanglish elements
        if any(s in response.lower() for s in self.pr_spanish_responses["spanglish"]):
            score += 0.2
            
        # Check sentence structure and flow
        if len(response.split()) >= 5 and "." in response:
            score += 0.2
            
        return min(score, 1.0)

    def score_sales_effectiveness(self, response: str) -> float:
        """Score sales effectiveness of response"""
        score = 0.0
        
        # Check for price discussion
        if any(term in response.lower() for term in ["precio", "costo", "deal", "discount", "$"]):
            score += 0.25
            
        # Check for call to action
        if any(term in response.lower() for term in ["¿", "?", "podemos", "quieres", "necesitas"]):
            score += 0.25
            
        # Check for value proposition
        if any(term in response.lower() for term in ["calidad", "mejor", "garantía", "servicio"]):
            score += 0.25
            
        # Check for urgency/closing
        if any(term in response.lower() for term in ["ahora", "hoy", "special", "limited"]):
            score += 0.25
            
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