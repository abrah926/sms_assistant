from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
from models import Message, MessageType
from config import HUGGINGFACE_TOKEN, LLM_MODEL_PATH, DEVICE
from sqlalchemy.ext.asyncio import AsyncSession
import re
import random

class MistralLLM:
    def __init__(self):
        print("Loading Mistral model and tokenizer...")
        if not HUGGINGFACE_TOKEN:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN not found in environment variables. "
                "Please add your token to .env file."
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=DEVICE,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_PATH,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Enhanced system prompt with more detailed guidelines
        self.system_prompt = """You are an expert metal sales agent with years of experience. Your role is to help customers purchase metals while maximizing sales opportunities.

PRODUCT INFORMATION:
1. Copper
   - Base price: $8,500/ton
   - Minimum order: 1 ton
   - Purity: 99.9%
   - Common uses: Electrical, construction

2. Steel
   - Base price: $800/ton
   - Minimum order: 5 tons
   - Grades available: Commercial, structural
   - Common uses: Construction, manufacturing

3. Aluminum
   - Base price: $2,400/ton
   - Minimum order: 2 tons
   - Grades: Industrial, commercial
   - Common uses: Manufacturing, aerospace

BULK DISCOUNTS:
- 200+ tons: 5% off
- 500+ tons: 10% off
- 1000+ tons: 15% off

SALES GUIDELINES:
1. Always acknowledge the customer's needs
2. Quote specific prices and minimum orders
3. Mention relevant bulk discounts
4. Ask closing questions
5. Keep responses concise (2-3 sentences)
6. Use professional, confident language

RESPONSE PATTERNS:
- Price inquiry: State price, minimum order, then ask about quantity
- Bulk order: Acknowledge order size, mention applicable discount, confirm order
- Technical question: Provide specification, relate to value, move to quantity
- Negotiation: Emphasize value, mention discounts, ask about volume

Remember: Always move the conversation towards closing a sale.
"""

        # Response templates for common scenarios
        self.response_templates = {
            "price_inquiry": [
                "{metal} is currently ${price}/ton with a {min_order} ton minimum. How many tons would you like to order?",
                "Our {metal} is priced at ${price}/ton, starting at {min_order} tons. Shall we discuss bulk discounts?",
                "The current rate for {metal} is ${price}/ton (minimum {min_order} tons). Would you like to place an order?"
            ],
            "bulk_order": [
                "Excellent choice! For {quantity} tons of {metal}, you qualify for a {discount}% discount, bringing the price to ${final_price}/ton. Shall we proceed with the order?",
                "Great volume! At {quantity} tons of {metal}, I can offer you a {discount}% discount (${final_price}/ton instead of ${price}/ton). Would you like to confirm this order?"
            ],
            "technical": [
                "Our {metal} meets industry standards with {purity}% purity, perfect for {use}. At ${price}/ton, how many tons do you need?",
                "The {metal} we supply is {purity}% pure, ideal for {use}. Would you like to discuss pricing for your required quantity?"
            ]
        }

    def _format_response(self, template_key: str, **kwargs) -> str:
        """Format response using templates"""
        templates = self.response_templates.get(template_key, [])
        if not templates:
            return None
        
        template = random.choice(templates)
        try:
            return template.format(**kwargs)
        except Exception as e:
            print(f"Error formatting response: {e}")
            return None

    async def generate(self, message: str, history: List[Message], db: AsyncSession) -> str:
        try:
            # Format conversation with system prompt and history
            conversation = f"{self.system_prompt}\n\nCustomer: {message}\nAgent:"
            
            # Generate response
            inputs = self.tokenizer(conversation, return_tensors="pt").to(DEVICE)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the agent's response
            response = response.split("Agent:")[-1].strip()
            
            # Try to use template if appropriate
            if "price" in message.lower() or "cost" in message.lower():
                metal = self._detect_metal(message)
                if metal:
                    template_response = self._format_response("price_inquiry",
                        metal=metal,
                        price=self._get_metal_price(metal),
                        min_order=self._get_min_order(metal)
                    )
                    if template_response:
                        return template_response
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?"

    def _detect_metal(self, message: str) -> str:
        """Detect mentioned metal in message"""
        message = message.lower()
        if "copper" in message:
            return "copper"
        elif "steel" in message:
            return "steel"
        elif "aluminum" in message or "aluminium" in message:
            return "aluminum"
        return None

    def _get_metal_price(self, metal: str) -> int:
        """Get base price for metal"""
        prices = {
            "copper": 8500,
            "steel": 800,
            "aluminum": 2400
        }
        return prices.get(metal)

    def _get_min_order(self, metal: str) -> int:
        """Get minimum order for metal"""
        min_orders = {
            "copper": 1,
            "steel": 5,
            "aluminum": 2
        }
        return min_orders.get(metal)

    def generate_sync(self, message: str, history: List[dict], db=None) -> str:
        """Synchronous version of generate for background tasks"""
        try:
            # Format conversation with system prompt and history
            conversation = f"{self.system_prompt}\n\nCustomer: {message}\nAgent:"
            
            # Generate response
            inputs = self.tokenizer(conversation, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the agent's response
            response = response.split("Agent:")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?"

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

    def generate_sync(self, prompt: str, context: str, db=None) -> str:
        """Synchronous version of generate"""
        try:
            print("\n=== LLM Generate Start ===")
            print(f"Prompt: {prompt}")
            print(f"Context: {context}")
            
            # Format conversation with clear context
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }]
            
            # Add context if available
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation context:\n{context}"
                })
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Debug print
            print(f"Generating response for: {prompt}")
            print(f"With context: {context}")

            # Generate response
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
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
                
                print(f"Raw response: {response}")
                
                # Only use fallbacks if response is empty or too short
                if not response or len(response.strip()) < 10:
                    if "steel" in prompt.lower():
                        return "Steel is $800/ton with a 5-ton minimum. For large orders over 200 tons, you get a 5% discount. Ready to place an order?"
                    elif "copper" in prompt.lower():
                        return "Copper is currently $8,500/ton, minimum order 1 ton. Would you like to discuss bulk pricing?"
                    elif "aluminum" in prompt.lower():
                        return "Our aluminum is $2,400/ton starting at 2 tons. How many tons do you need?"
                    else:
                        return "We offer steel, copper, and aluminum at competitive prices. Which metal interests you today?"
                
                print(f"Raw LLM output: {response[:200]}...")
                print("=== LLM Generate End ===\n")
                return response.strip()
                
            except Exception as e:
                print(f"Error cleaning response: {e}")
                return self._get_smart_fallback(prompt)

        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return self._get_smart_fallback(prompt)

    def _get_smart_fallback(self, prompt: str) -> str:
        """Get context-aware fallback response"""
        prompt_lower = prompt.lower()
        if "price" in prompt_lower or "cost" in prompt_lower:
            return "Our current prices are: steel at $800/ton, copper at $8,500/ton, and aluminum at $2,400/ton. Which interests you?"
        elif "order" in prompt_lower or "buy" in prompt_lower:
            return "I can help you place an order. We offer bulk discounts starting at 200 tons. What metal and quantity are you interested in?"
        elif any(metal in prompt_lower for metal in ["steel", "copper", "aluminum"]):
            return "I can provide pricing and availability for that metal. Would you like to know our current rates?"
        else:
            return "We specialize in steel, copper, and aluminum. What metal are you interested in today?"

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

    def clean_response(self, response: str) -> str:
        """Clean model response"""
        try:
            # Remove system prompts and special tokens
            response = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', response, flags=re.DOTALL)
            response = re.sub(r'<\|.*?\|>', '', response)
            
            # Remove any gibberish patterns
            response = re.sub(r'[\'\"]+[A-Z&]+[\'\"]+', '', response)
            response = re.sub(r'[\'\"]<+[\'\"]', '', response)
            
            # Clean up the text
            response = re.sub(r'[^a-zA-Z0-9\s.,!?$%()-]', '', response)
            response = ' '.join(response.split())
            
            # If response is too short or gibberish, use template
            if len(response.strip()) < 20 or not re.match(r'^[a-zA-Z0-9\s.,!?$%()-]+$', response):
                return self._get_template_response()
            
            return response.strip()
        except Exception as e:
            print(f"Error cleaning response: {e}")
            return self._get_template_response()

    def _get_template_response(self) -> str:
        """Get a template response for copper price inquiry"""
        return "Copper is currently $8,500/ton with a minimum order of 1 ton. Would you like to place an order?"