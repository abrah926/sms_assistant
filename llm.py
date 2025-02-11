from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
from models import Message, MessageType
from config import HUGGINGFACE_TOKEN, LLM_MODEL_PATH, DEVICE
from sqlalchemy.ext.asyncio import AsyncSession
import re

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
            
            # Define improved system prompt
            self.system_prompt = """You are a professional SMS sales assistant for a metal company. Be concise and natural in your responses.

Available Metals:
- Steel: $800/ton (min: 5 tons)
- Copper: $8,500/ton (min: 1 ton)
- Aluminum: $2,400/ton (min: 2 tons)

Guidelines:
- Write like a human SMS, be brief and clear
- Always mention minimum order quantities with prices
- Be friendly but professional
- No special characters or formatting
- Keep responses under 160 characters when possible"""
            
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
            # Prepare conversation with clear instructions
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }, {
                "role": "user",
                "content": prompt
            }]

            # Generate response
            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,  # Shorter for SMS-like responses
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and clean the actual response
            try:
                # Remove system prompt and template markers
                response = response.replace(self.system_prompt, "")
                response = re.sub(r'<\|.*?\|>', '', response)
                
                # Extract the actual response (after user message)
                if prompt in response:
                    response = response.split(prompt)[-1]
                
                # Clean up the text
                response = re.sub(r'[^a-zA-Z0-9\s.,!?$%()-]', '', response)
                response = ' '.join(response.split())
                
                # If response is invalid, use appropriate fallback
                if not response or len(response.strip()) < 10:
                    return "Copper is $8,500/ton with a 1 ton minimum. Would you like to place an order?"
                
                return response.strip()
                
            except Exception as e:
                print(f"Error cleaning response: {e}")
                return "Copper is $8,500/ton with a 1 ton minimum. Would you like to place an order?"

        except Exception as e:
            print(f"Error in generate_sync: {str(e)}")
            return "Hi! I can help you with metal pricing and orders. What metal are you interested in?"

    def _clean_llm_response(self, response: str, prompt: str) -> str:
        """Clean LLM response to look more human"""
        try:
            # Remove system prompt and template markers
            markers = [
                "<|im_start|>", "<|im_end|>",
                "system", "user", "assistant",
                self.system_prompt,
                prompt
            ]
            
            for marker in markers:
                response = response.replace(marker, "")
            
            # Remove any remaining special characters
            response = re.sub(r'[^a-zA-Z0-9\s.,!?$%()-]', '', response)
            
            # Remove multiple spaces and newlines
            response = ' '.join(response.split())
            
            # Remove common AI prefixes
            prefixes = ['AI:', 'Assistant:', 'Bot:', 'Response:']
            for prefix in prefixes:
                if response.lower().startswith(prefix.lower()):
                    response = response[len(prefix):].strip()
            
            # Format currency consistently
            response = re.sub(r'\$\s+', '$', response)
            
            # Add professional closing if missing
            if not any(x in response.lower() for x in ['thank', 'help', 'assist']):
                response += " Let me know if you need any other information."
            
            return response.strip()
        
        except Exception as e:
            print(f"Error cleaning response: {e}")
            return "The price of copper is $8,500 per ton with a minimum order of 1 ton. Would you like to place an order?"