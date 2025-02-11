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
            
            # Improved system prompt focused on sales
            self.system_prompt = """You are a professional sales assistant for a metal trading company. Your primary goal is to sell metal products and provide excellent customer service.

Key Information:
- Products and Pricing:
  * Copper: $8,500/ton (minimum: 1 ton)
  * Steel: $800/ton (minimum: 5 tons)
  * Aluminum: $2,400/ton (minimum: 2 tons)

Guidelines:
1. Always be sales-focused but professional
2. Respond in a clear, concise manner
3. Always mention pricing with minimum order quantities
4. Offer bulk discounts for orders over:
   - 200 tons: 5% discount
   - 500 tons: 10% discount
   - 1000 tons: 15% discount
5. Use natural, human-like language
6. Keep responses under 2-3 sentences
7. Always encourage the sale

Example Responses:
- "Copper is $8,500 per ton with a 1 ton minimum. Would you like to place an order?"
- "I can offer steel at $800/ton, minimum 5 tons. For larger orders over 200 tons, you'll get a 5% discount."
- "Our aluminum is $2,400/ton with a 2-ton minimum order. How many tons would you like to purchase?"
"""
            
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
            # Format conversation with clear context
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }]
            
            # Add relevant history if available
            if history:
                for msg in history[-3:]:  # Only last 3 messages for context
                    messages.append({
                        "role": "user" if msg.direction == "incoming" else "assistant",
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
                    max_new_tokens=100,  # Shorter responses
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
                
                # Validate response
                if not response or len(response.strip()) < 10:
                    return get_fallback_response(prompt)
                
                return response.strip()
                
            except Exception as e:
                print(f"Error cleaning response: {e}")
                return get_fallback_response(prompt)

        except Exception as e:
            print(f"Error in generate_sync: {str(e)}")
            return "I apologize, but I can help you with current metal pricing and orders. What metal are you interested in?"

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