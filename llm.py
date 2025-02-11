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
            
            # Define system prompt
            self.system_prompt = """You are a professional metal sales assistant. 
            Your role is to help customers with metal purchases, pricing, and general inquiries.
            Be concise, professional, and business-focused.
            
            Available metals and current prices:
            - Steel: $800/ton
            - Copper: $8,500/ton
            - Aluminum: $2,400/ton
            
            Minimum order quantities:
            - Steel: 5 tons
            - Copper: 1 ton
            - Aluminum: 2 tons
            
            Always mention minimum order quantities when discussing prices."""
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise

    async def generate(self, prompt: str, history: List[Message], db: AsyncSession = None) -> str:
        """Generate response with improved context handling"""
        try:
            # Format conversation
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }]
            
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

            print(f"Generating response for prompt: {prompt}")
            print(f"With {len(history)} messages in history")

            # Generate response
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,  # Increased for better responses
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the actual response (after the last user message)
            parts = response.split(prompt)
            if len(parts) > 1:
                response = parts[-1]
            
            # Clean up
            response = response.replace("Assistant:", "").strip()
            
            print(f"Generated response: {response[:100]}...")
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

    def generate_sync(self, prompt: str, history: List[Message], db=None) -> str:
        """Synchronous version of generate"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": prompt})

            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace("Assistant:", "").strip()
        except Exception as e:
            print(f"Error in generate_sync: {str(e)}")
            return "I apologize, I'm having trouble processing your request." 