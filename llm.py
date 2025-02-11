from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
from models import Message, MessageType
from config import HUGGINGFACE_TOKEN, LLM_MODEL_PATH, DEVICE
from sqlalchemy.ext.asyncio import AsyncSession

class MistralLLM:
    def __init__(self):
        print("Initializing LLM...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_PATH,
                token=HUGGINGFACE_TOKEN
            )
            print("Tokenizer initialized")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PATH,
                token=HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16
            ).to(DEVICE)
            print("Model initialized")
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise

    async def generate(self, prompt: str, history: List[Message], db: AsyncSession) -> str:
        """Simple synchronous generation without complex async patterns"""
        try:
            # Format messages
            messages = []
            messages.append({
                "role": "system", 
                "content": "You are a professional metal sales assistant. Be concise and business-focused."
            })
            
            # Add history
            for msg in history:
                role = "user" if msg.direction == "incoming" else "assistant"
                messages.append({
                    "role": role,
                    "content": msg.content
                })
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Generate response
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = response.replace("Assistant:", "").strip()
            return response

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            print(f"Error type: {type(e)}")
            return "I apologize, I'm having trouble processing your request. How can I help you today?"

    def _format_history(self, history: List[Message]) -> str:
        formatted = []
        for msg in history:
            prefix = "Assistant: " if msg.direction == "outgoing" else "User: "
            formatted.append(f"{prefix}{msg.content}")
        return "\n".join(formatted)
    
    async def _process_response(self, response: str, db: AsyncSession) -> str:
        # Implement the logic to process the response and return the final cleaned response
        # This is a placeholder and should be replaced with the actual implementation
        return response

    def _clean_response(self, response: str, prompt: str) -> str:
        # Remove the prompt from the response
        clean = response[len(prompt):].strip()
        # Remove any additional "Assistant:" prefixes
        clean = clean.replace("Assistant:", "").strip()
        return clean 