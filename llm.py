from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
from models import Message
from config import HUGGINGFACE_TOKEN, LLM_MODEL_PATH, DEVICE
from business_functions import get_product_price, check_inventory, process_order
from sqlalchemy.ext.asyncio import AsyncSession

class MistralLLM:
    def __init__(self):
        print("Initializing LLM...")  # Debug print
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_PATH,
                token=HUGGINGFACE_TOKEN
            )
            print("Tokenizer initialized")  # Debug print
            
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PATH,
                token=HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16
            ).to(DEVICE)
            print("Model initialized")  # Debug print
            
            # Define business functions
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_product_price",
                        "description": "Get product price with bulk discounts",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "product_name": {
                                    "type": "string",
                                    "description": "Name of the metal product"
                                },
                                "quantity_kg": {
                                    "type": "number",
                                    "description": "Quantity in kilograms"
                                }
                            },
                            "required": ["product_name", "quantity_kg"]
                        }
                    }
                },
                # Add other function definitions...
            ]
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise
        
    async def generate(self, prompt: str, history: List[Message], db: AsyncSession, max_length: int = 150) -> str:
        context = self._format_history(history)
        messages = [
            {"role": "system", "content": "You are a professional metal sales assistant. Be concise and business-focused."},
            *[{"role": "user" if msg.direction == "incoming" else "assistant", "content": msg.content} for msg in history],
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return await self._process_response(response, db)
    
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