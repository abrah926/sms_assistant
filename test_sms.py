import asyncio
from httpx import AsyncClient
from datetime import datetime
from typing import List, Dict
import json

class TestScenario:
    def __init__(self, name: str, customer: Dict, messages: List[Dict[str, str]]):
        self.name = name
        self.customer = customer
        self.messages = messages

# Test data
existing_customer = {
    "name": "Abraham",
    "phone": "+1234567890",
    "has_payment": True
}

new_customer = {
    "name": "Sarah",
    "phone": "+1987654321",
    "has_payment": False
}

scenarios = [
    TestScenario(
        "New Customer Service Inquiry",
        new_customer,
        [
            {"text": "Hi, I'm interested in buying metals from your company"},
            {"text": "What types of metal do you sell and what are your prices?"},
            {"text": "Do you offer bulk discounts?"},
            {"text": "What's the minimum order quantity for steel?"}
        ]
    ),
    
    TestScenario(
        "Bulk Order - Existing Customer",
        existing_customer,
        [
            {"text": "Hi, this is Abraham. I need a bulk order of copper"},
            {"text": "I need about 500kg for a new project"},
            {"text": "Yes, I'll use my saved payment method"},
            {"text": "Can you confirm the total with the bulk discount?"}
        ]
    ),
    
    TestScenario(
        "Inventory Check",
        existing_customer,
        [
            {"text": "Do you have aluminium in stock?"},
            {"text": "What's the current price per kg?"},
            {"text": "And how much is available right now?"},
            {"text": "Great, I'll think about it and get back to you"}
        ]
    ),
    
    TestScenario(
        "New Customer Order",
        new_customer,
        [
            {"text": "I want to order 100kg of steel"},
            {"text": "I'm a new customer, my name is Sarah"},
            {"text": "Here's my credit card: 4111-1111-1111-1111 exp 12/25"},
            {"text": "Yes, please process the order now"}
        ]
    )
]

async def test_conversation(scenario: TestScenario):
    async with AsyncClient() as client:
        print(f"\nTesting Scenario: {scenario.name}")
        print(f"Customer: {scenario.customer['name']}")
        print("=" * 50)
        
        for msg in scenario.messages:
            response = await client.post(
                "http://localhost:8000/message/webhook",
                json={
                    "from_number": scenario.customer["phone"],
                    "body": msg["text"]
                }
            )
            
            # Add debug information
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response Content: {response.content}")
            
            if response.status_code != 200:
                print(f"Error Response: {response.text}")
                continue
                
            try:
                result = response.json()
                print(f"\nUser: {msg['text']}")
                print(f"AI: {result['response']}")
                print(f"Type: {result['type']}")
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                print(f"Raw response: {response.text}")
            
            await asyncio.sleep(1)

async def run_all_tests():
    print("Starting Metal Sales SMS Tests")
    print("=" * 50)
    
    for scenario in scenarios:
        await test_conversation(scenario)
        print("\nPress Enter to continue to next scenario...")
        input()

if __name__ == "__main__":
    print("Ensure your FastAPI server is running (uvicorn app:app --reload)")
    print("Make sure the database is populated with test products")
    asyncio.run(run_all_tests()) 