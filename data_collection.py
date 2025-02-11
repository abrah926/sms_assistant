from models import TrainingExample, Message, MessageType
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.sql import text
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import List, Dict, Tuple
import csv
import requests
from bs4 import BeautifulSoup
import re
import asyncio
import aiohttp
import random
from itertools import product as itertools_product
from datasets import load_dataset

async def store_training_example(
    session: AsyncSession,
    customer_message: str,
    agent_response: str,
    context: str = None,
    metadata: dict = None
):
    """Store a new training example with deduplication"""
    try:
        # Hash the message pair to check for duplicates quickly
        message_hash = hash(f"{customer_message}:{agent_response}")
        
        # Use a class-level set to track seen messages
        if not hasattr(store_training_example, 'seen_messages'):
            store_training_example.seen_messages = set()
            
        # Check if we've seen this message pair before
        if message_hash in store_training_example.seen_messages:
            return None
            
        # Add to seen messages
        store_training_example.seen_messages.add(message_hash)
        
        # Create new example
        example = TrainingExample(
            customer_message=customer_message,
            agent_response=agent_response,
            context=context,
            meta_info=metadata or {}
        )
        session.add(example)
        await session.commit()
        return example
        
    except Exception as e:
        print(f"Error storing training example: {e}")
        await session.rollback()
        return None

# Example training data
INITIAL_TRAINING_DATA = [
    # Price inquiries with variations
    {
        "customer_message": "How much is copper per ton?",
        "agent_response": "Copper is currently $8,500/ton with a 1 ton minimum. Would you like to place an order?",
        "metadata": {"product": "copper", "intent": "price_inquiry"}
    },
    {
        "customer_message": "What's your copper pricing?",
        "agent_response": "Our copper is $8,500 per ton, minimum order 1 ton. For orders over 200 tons, you'll get a 5% discount. How many tons would you like?",
        "metadata": {"product": "copper", "intent": "price_inquiry"}
    },
    
    # Negotiation scenarios
    {
        "customer_message": "That's a bit expensive. Can you do better?",
        "agent_response": "I understand. For larger orders, we offer significant discounts: 5% off for 200+ tons, 10% off for 500+ tons. What volume are you considering?",
        "metadata": {"intent": "negotiation", "stage": "price_objection"}
    },
    {
        "customer_message": "I'm comparing prices with other suppliers",
        "agent_response": "We offer the best value with our quality and reliability. Plus, our bulk discounts can save you significantly. What quantities are you looking at?",
        "metadata": {"intent": "negotiation", "stage": "competition"}
    },
    
    # Complex inquiries
    {
        "customer_message": "Need 300 tons of steel and 100 tons of copper",
        "agent_response": "Great! For 300 tons of steel, you'll get 5% off ($760/ton), and copper at $8,500/ton. Total would be $1,078,000. Shall we proceed with this order?",
        "metadata": {"intent": "bulk_order", "products": ["steel", "copper"], "quantities": [300, 100]}
    },
    
    # Urgency handling
    {
        "customer_message": "How quickly can you deliver 50 tons of aluminum?",
        "agent_response": "We can arrange delivery for 50 tons of aluminum ($2,400/ton) within the week. Would you like to secure this order now?",
        "metadata": {"product": "aluminum", "intent": "delivery_inquiry", "quantity": 50}
    },
    
    # Follow-up scenarios
    {
        "customer_message": "Let me think about it",
        "agent_response": "I understand. Keep in mind these prices are subject to market changes. Would you like me to hold this quote for you for 24 hours?",
        "metadata": {"intent": "follow_up", "stage": "hesitation"}
    },
    
    # Technical questions
    {
        "customer_message": "What's the purity of your copper?",
        "agent_response": "Our copper is 99.9% pure, meeting industry standards. At $8,500/ton, it's excellent value. How many tons would you need?",
        "metadata": {"product": "copper", "intent": "technical_inquiry"}
    },
    
    # Payment and terms
    {
        "customer_message": "What are your payment terms?",
        "agent_response": "We offer net-30 terms for established customers. For your first order, we require 50% deposit. Would you like to discuss specific quantities?",
        "metadata": {"intent": "payment_inquiry"}
    },
    
    # Closing scenarios
    {
        "customer_message": "Yes, I'm interested in the steel",
        "agent_response": "Excellent! How many tons of steel should I quote for you? Remember, orders over 200 tons get a 5% discount.",
        "metadata": {"product": "steel", "intent": "interest", "stage": "closing"}
    },
    
    # Handling specific requirements
    {
        "customer_message": "Do you deliver to Chicago?",
        "agent_response": "Yes, we deliver to Chicago! Let's get your order details first - which metal and quantity are you interested in?",
        "metadata": {"intent": "logistics_inquiry", "location": "Chicago"}
    },
    
    # Market information
    {
        "customer_message": "Are prices going up soon?",
        "agent_response": "Market trends suggest possible increases. I recommend securing your order now at current rates. What quantity would you like to lock in?",
        "metadata": {"intent": "market_inquiry", "stage": "urgency_creation"}
    }
]

async def initialize_training_data(session: AsyncSession):
    """Initialize database with some training examples"""
    for example in INITIAL_TRAINING_DATA:
        await store_training_example(
            session,
            example["customer_message"],
            example["agent_response"],
            metadata=example["metadata"]
        )

async def collect_successful_conversations(session: AsyncSession):
    """Collect successful conversations from message history"""
    
    # Updated query to prevent duplicates
    query = text("""
        WITH conversation_pairs AS (
            SELECT DISTINCT ON (m1.phone, m1.content)  -- Add DISTINCT ON to get unique conversations
                m1.phone,
                m1.content as customer_message,
                FIRST_VALUE(m2.content) OVER (
                    PARTITION BY m1.phone, m1.content 
                    ORDER BY m2.timestamp
                ) as agent_response,
                m1.timestamp,
                m1.meta_data as customer_metadata,
                m2.meta_data as agent_metadata
            FROM messages m1
            JOIN messages m2 ON m1.phone = m2.phone
            WHERE 
                m1.direction = 'incoming'
                AND m2.direction = 'outgoing'
                AND m2.timestamp > m1.timestamp
                AND m2.timestamp <= m1.timestamp + interval '2 minutes'
                AND (
                    m2.meta_data->>'processed' = 'true'
                    OR m2.message_type = 'ORDER'
                    OR m2.content ILIKE '%thank%'
                    OR m2.content ILIKE '%great%'
                    OR m2.content ILIKE '%excellent%'
                )
        )
        SELECT * FROM conversation_pairs
        ORDER BY timestamp DESC
        LIMIT 100
    """)
    
    result = await session.execute(query)
    conversations = result.fetchall()
    
    # Store successful conversations as training examples
    for conv in conversations:
        # Extract intent and product from messages
        metadata = {
            "intent": detect_intent(conv.customer_message),
            "product": detect_product(conv.customer_message),
            "source": "successful_conversation",
            "timestamp": conv.timestamp.isoformat(),
            "customer_metadata": conv.customer_metadata,
            "agent_metadata": conv.agent_metadata
        }
        
        await store_training_example(
            session,
            conv.customer_message,
            conv.agent_response,
            metadata=metadata
        )
    
    return len(conversations)

def detect_intent(message: str) -> str:
    """Detect the intent of a message"""
    message = message.lower()
    
    if any(word in message for word in ["price", "cost", "how much"]):
        return "price_inquiry"
    elif any(word in message for word in ["minimum", "min", "least"]):
        return "min_order_inquiry"
    elif any(word in message for word in ["discount", "cheaper", "better price"]):
        return "discount_inquiry"
    elif any(word in message for word in ["delivery", "ship", "when"]):
        return "delivery_inquiry"
    elif any(word in message for word in ["stock", "available", "have"]):
        return "availability_check"
    elif any(word in message for word in ["buy", "order", "purchase"]):
        return "bulk_order"
    else:
        return "general_inquiry"

def detect_product(message: str) -> str:
    """Detect the product mentioned in a message"""
    message = message.lower()
    
    if "copper" in message:
        return "copper"
    elif "steel" in message:
        return "steel"
    elif "aluminum" in message or "aluminium" in message:
        return "aluminum"
    else:
        return None

# Add function to periodically collect training data
async def schedule_training_collection(session: AsyncSession):
    """Schedule periodic collection of training data"""
    try:
        # Collect successful conversations from the last 24 hours
        count = await collect_successful_conversations(session)
        print(f"Collected {count} new training examples")
        
        # You could also add other collection methods here
        
    except Exception as e:
        print(f"Error collecting training data: {e}")

async def import_external_sales_data(
    session: AsyncSession,
    source_type: str,
    file_path: str,
    mapping: Dict = None
):
    """Import sales conversations from external sources"""
    try:
        if source_type == "json":
            data = import_json_conversations(file_path, mapping)
        elif source_type == "excel":
            data = import_excel_conversations(file_path, mapping)
        elif source_type == "txt":
            data = import_text_conversations(file_path, mapping)
        elif source_type == "xml":
            data = import_xml_conversations(file_path, mapping)
        elif source_type == "hubspot":
            data = import_hubspot_conversations(file_path, mapping)
        elif source_type == "salesforce":
            data = import_salesforce_conversations(file_path, mapping)
        elif source_type == "zendesk":
            data = import_zendesk_conversations(file_path, mapping)
        elif source_type == "csv":
            data = import_csv_conversations(file_path, mapping)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Store imported conversations
        for conversation in data:
            await store_training_example(
                session,
                conversation["customer_message"],
                conversation["agent_response"],
                metadata={
                    "source": source_type,
                    "import_date": datetime.now().isoformat(),
                    "original_metadata": conversation.get("metadata", {}),
                    "success_metrics": conversation.get("success_metrics", {})
                }
            )
        
        return len(data)

    except Exception as e:
        print(f"Error importing data: {e}")
        return 0

def import_hubspot_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from HubSpot export"""
    conversations = []
    
    # Default field mapping
    field_map = mapping or {
        "customer_message": "contact_message",
        "agent_response": "agent_reply",
        "timestamp": "timestamp",
        "deal_stage": "deal_stage",
        "deal_amount": "amount",
        "success": "closed_won"
    }
    
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        try:
            # Only import successful sales interactions
            if row[field_map["success"]] == True:
                conversations.append({
                    "customer_message": row[field_map["customer_message"]],
                    "agent_response": row[field_map["agent_response"]],
                    "metadata": {
                        "deal_stage": row[field_map["deal_stage"]],
                        "deal_amount": row[field_map["deal_amount"]],
                        "timestamp": row[field_map["timestamp"]]
                    },
                    "success_metrics": {
                        "converted": True,
                        "deal_value": row[field_map["deal_amount"]]
                    }
                })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return conversations

def import_salesforce_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from Salesforce export"""
    conversations = []
    
    # Default Salesforce field mapping
    field_map = mapping or {
        "customer_message": "Description",
        "agent_response": "Comments",
        "opportunity_stage": "StageName",
        "amount": "Amount",
        "closed_won": "IsWon"
    }
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
        for record in data:
            try:
                if record[field_map["closed_won"]]:
                    conversations.append({
                        "customer_message": record[field_map["customer_message"]],
                        "agent_response": record[field_map["agent_response"]],
                        "metadata": {
                            "opportunity_stage": record[field_map["opportunity_stage"]],
                            "amount": record[field_map["amount"]],
                            "source": "salesforce"
                        },
                        "success_metrics": {
                            "converted": True,
                            "deal_value": record[field_map["amount"]]
                        }
                    })
            except Exception as e:
                print(f"Error processing Salesforce record: {e}")
                continue
    
    return conversations

def import_csv_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from generic CSV format"""
    conversations = []
    
    # Default CSV field mapping
    field_map = mapping or {
        "customer_message": "customer_message",
        "agent_response": "agent_response",
        "success": "converted",
        "product": "product",
        "quantity": "quantity",
        "deal_value": "value"
    }
    
    try:
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            try:
                conversations.append({
                    "customer_message": row[field_map["customer_message"]],
                    "agent_response": row[field_map["agent_response"]],
                    "metadata": {
                        "product": row.get(field_map["product"]),
                        "quantity": row.get(field_map["quantity"]),
                        "source": "csv_import"
                    },
                    "success_metrics": {
                        "converted": row.get(field_map["success"], False),
                        "deal_value": row.get(field_map["deal_value"], 0)
                    }
                })
            except Exception as e:
                print(f"Error processing CSV row: {e}")
                continue
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    return conversations

def import_zendesk_conversations(file_path: str, mapping: dict) -> List[Dict]:
    """Import conversations from Zendesk export"""
    conversations = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for ticket in data['tickets']:
                if 'conversations' in ticket:
                    for conv in ticket['conversations']:
                        customer_msg = conv.get(mapping['customer_message'], '')
                        agent_msg = conv.get(mapping['agent_response'], '')
                        if customer_msg and agent_msg:
                            conversations.append({
                                'customer_message': customer_msg,
                                'agent_response': agent_msg,
                                'source': 'zendesk'
                            })
    except Exception as e:
        print(f"Error importing Zendesk data: {e}")
    return conversations

async def import_huggingface_datasets(session: AsyncSession) -> int:
    """Import and combine multiple relevant datasets"""
    try:
        print("Loading HuggingFace datasets...")
        stored_count = 0
        
        # Load multiple relevant datasets with trust_remote_code=True
        datasets = {
            "daily_dialog": load_dataset("daily_dialog", split="train", trust_remote_code=True),
            "blended_skill_talk": load_dataset("blended_skill_talk", split="train", trust_remote_code=True),
            "multi_woz_v22": load_dataset("multi_woz_v22", split="train", trust_remote_code=True)
        }
        
        for dataset_name, dataset in datasets.items():
            print(f"Processing {dataset_name}...")
            
            # Extract conversations based on dataset structure
            if dataset_name == "daily_dialog":
                for item in dataset:
                    # Daily Dialog alternates between speakers
                    for i in range(0, len(item['dialog']) - 1, 2):
                        try:
                            await store_training_example(
                                session,
                                customer_message=item['dialog'][i],
                                agent_response=item['dialog'][i + 1],
                                metadata={
                                    "source": f"huggingface_{dataset_name}",
                                    "emotion": item['emotion'][i],
                                    "act": item['act'][i]
                                }
                            )
                            stored_count += 1
                        except Exception as e:
                            print(f"Error storing conversation: {e}")
                            continue
                            
            elif dataset_name == "blended_skill_talk":
                for item in dataset:
                    try:
                        await store_training_example(
                            session,
                            customer_message=item['previous_utterance'],
                            agent_response=item['human_response'],
                            metadata={
                                "source": f"huggingface_{dataset_name}",
                                "context": item.get('context', '')
                            }
                        )
                        stored_count += 1
                    except Exception as e:
                        print(f"Error storing conversation: {e}")
                        continue
            
            elif dataset_name == "multi_woz_v22":
                for item in dataset:
                    try:
                        # Extract customer-agent pairs from dialogue
                        turns = item['turns']
                        for i in range(0, len(turns) - 1, 2):
                            if turns[i]['speaker'] == 'USER':
                                await store_training_example(
                                    session,
                                    customer_message=turns[i]['utterance'],
                                    agent_response=turns[i + 1]['utterance'],
                                    metadata={
                                        "source": f"huggingface_{dataset_name}",
                                        "domain": item.get('domains', []),
                                        "dialogue_id": item['dialogue_id']
                                    }
                                )
                                stored_count += 1
                    except Exception as e:
                        print(f"Error storing conversation: {e}")
                        continue
        
        print(f"Successfully imported {stored_count} conversations from HuggingFace datasets")
        return stored_count
        
    except Exception as e:
        print(f"Error importing HuggingFace datasets: {e}")
        return 0

def import_excel_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from Excel file"""
    conversations = []
    
    # Default Excel field mapping
    field_map = mapping or {
        "customer_message": "customer_message",
        "agent_response": "agent_response",
        "timestamp": "timestamp",
        "product": "product",
        "success": "converted"
    }
    
    try:
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            conversations.append({
                "customer_message": row[field_map["customer_message"]],
                "agent_response": row[field_map["agent_response"]],
                "metadata": {
                    "source": "excel_import",
                    "timestamp": row.get(field_map["timestamp"]),
                    "product": row.get(field_map["product"])
                }
            })
    except Exception as e:
        print(f"Error reading Excel file: {e}")
    
    return conversations

def import_json_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from JSON file"""
    conversations = []
    
    # Default JSON field mapping
    field_map = mapping or {
        "customer_message": "customer_message",
        "agent_response": "agent_response",
        "metadata": "metadata"
    }
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                conversations.append({
                    "customer_message": item[field_map["customer_message"]],
                    "agent_response": item[field_map["agent_response"]],
                    "metadata": item.get(field_map["metadata"], {})
                })
    except Exception as e:
        print(f"Error reading JSON file: {e}")
    
    return conversations

def import_text_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from text file"""
    conversations = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Extract conversations using existing patterns
            for pattern in [
                r"Customer:\s*(.*?)\s*Agent:\s*(.*?)(?=\n*Customer:|$)",
                r"Client:\s*(.*?)\s*Sales:\s*(.*?)(?=\n*Client:|$)"
            ]:
                matches = re.finditer(pattern, content, re.DOTALL)
                for match in matches:
                    conversations.append({
                        "customer_message": match.group(1).strip(),
                        "agent_response": match.group(2).strip(),
                        "metadata": {"source": "text_import"}
                    })
    except Exception as e:
        print(f"Error reading text file: {e}")
    
    return conversations

def import_xml_conversations(file_path: str, mapping: Dict = None) -> List[Dict]:
    """Import conversations from XML file"""
    conversations = []
    
    try:
        from xml.etree import ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for conv in root.findall('.//conversation'):
            customer_msg = conv.find('customer').text.strip()
            agent_msg = conv.find('agent').text.strip()
            conversations.append({
                "customer_message": customer_msg,
                "agent_response": agent_msg,
                "metadata": {"source": "xml_import"}
            })
    except Exception as e:
        print(f"Error reading XML file: {e}")
    
    return conversations

class SalesDataCollector:
    def __init__(self):
        self.sources = {
            "sales_hacker": "https://www.saleshacker.com/",
            "hubspot_blog": "https://blog.hubspot.com/sales/",
            "close_blog": "https://blog.close.com/"
        }
        
        self.conversation_patterns = [
            r"Customer:\s*(.*?)\s*Sales(?:person|rep):\s*(.*?)(?=Customer:|$)",
            r"Prospect:\s*(.*?)\s*Rep:\s*(.*?)(?=Prospect:|$)",
            r"Client:\s*(.*?)\s*Agent:\s*(.*?)(?=Client:|$)"
        ]
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_customer_length": 10,
            "min_agent_length": 20,
            "max_message_length": 500,
            "min_relevance_score": 0.7
        }
        
        # Keywords for relevance checking
        self.relevant_keywords = {
            "products": ["metal", "steel", "copper", "aluminum", "tonnes", "tons"],
            "sales_terms": ["price", "cost", "order", "delivery", "quantity", "discount"],
            "business_terms": ["quote", "purchase", "supply", "contract", "terms"]
        }
        
        # Patterns for better conversation extraction
        self.conversation_patterns.extend([
            # Add more sophisticated patterns
            r"(?:Customer|Client|Buyer)[\s>:\-]+([^>:\n]+?)[\s>:\-]+(?:Sales|Agent|Rep)[\s>:\-]+([^>:\n]+)",
            r"(?:Q|Question)[\s>:\-]+([^>:\n]+?)[\s>:\-]+(?:A|Answer)[\s>:\-]+([^>:\n]+)",
            r"(?:Inquiry|Request)[\s>:\-]+([^>:\n]+?)[\s>:\-]+(?:Response|Reply)[\s>:\-]+([^>:\n]+)"
        ])
        
        # Add variation templates
        self.variations = {
            "price_inquiry": [
                "How much is {product}?",
                "What's the price of {product}?",
                "Can you tell me the cost of {product}?",
                "I need pricing for {product}",
                "What are your {product} rates?",
                "How much would {product} cost?",
                "What's your {product} pricing?",
                "Give me a quote for {product}"
            ],
            "quantity_variations": [
                "100", "200", "300", "500", "1000", "1500", "2000", "5000"
            ],
            "products": [
                "copper", "steel", "aluminum", "iron", "brass", "bronze", "titanium"
            ],
            "units": [
                "tons", "kilos", "kg", "metric tons"
            ]
        }
        
        # Price ranges for products
        self.product_prices = {
            "copper": (8000, 9000),
            "steel": (700, 900),
            "aluminum": (2200, 2600),
            "iron": (500, 700),
            "brass": (4000, 4500),
            "bronze": (4500, 5000),
            "titanium": (15000, 17000)
        }
        
        # Expand response templates
        self.response_templates = {
            "bulk_order": [
                "Excellent! For {quantity} tons of {product}, I can offer a {discount}% discount (${final_price}/ton). Would you like to proceed?",
                "Great volume! At {quantity} tons, you qualify for our {discount}% bulk discount, making it ${final_price}/ton. Shall we process this order?",
                "Perfect! I can offer you {quantity} tons of {product} at ${final_price}/ton (includes {discount}% bulk discount). Ready to move forward?",
                "That's a substantial order! I'll give you a {discount}% discount on {quantity} tons, bringing it to ${final_price}/ton. Should I prepare the paperwork?",
                "For an order of {quantity} tons, you'll get our {discount}% volume discount. Final price: ${final_price}/ton. Would you like delivery details?"
            ],
            "price_inquiry": [
                "{product} is ${price}/ton with a {min_order} ton minimum. Interested in placing an order?",
                "Current {product} pricing is ${price}/ton (min: {min_order} tons). Shall we discuss bulk discounts?",
                "We offer {product} at ${price}/ton, starting from {min_order} tons. How much do you need?",
                "The market rate for {product} is ${price}/ton. Orders start at {min_order} tons. Would you like a formal quote?",
                "I can supply {product} at ${price}/ton. Minimum order is {min_order} tons. Interested in learning about our bulk discounts?"
            ],
            "negotiation": [
                "I understand price is important. For orders over {threshold} tons, I can offer a {discount}% discount. What volume are you considering?",
                "While our base price is ${price}/ton, we offer significant discounts for bulk orders. What quantity do you need?",
                "Let me see what I can do. If you can commit to {threshold} tons, I'll give you a {discount}% discount. Interested?",
                "Our pricing is competitive when you consider our quality and reliability. Plus, I can offer {discount}% off for large orders.",
                "How about this: order {threshold} tons and I'll give you a {discount}% discount. That would bring it down to ${discounted_price}/ton."
            ]
        }

        # Add this after the existing response_templates
        self.advanced_templates = {
            "objection_handling": [
                "I understand your concern about {objection}. Many clients initially felt the same, but they found our {benefit} more than compensates. For example, {example}.",
                "That's a fair point about {objection}. Let me share how other customers have addressed this: {solution}. Would that work for you?",
                "I hear you on {objection}. What if I could show you how {solution} could actually save you money in the long run?"
            ],
            "competitive_positioning": [
                "While others might offer lower prices, our {quality_feature} ensures you save {savings}% on waste. Would you like to see the calculations?",
                "What sets us apart is {unique_feature}. This means {benefit} for your operations. How would that impact your business?",
                "Our {product} has {certification} certification, guaranteeing {quality_level}. How important is quality assurance to your process?"
            ],
            "urgency_creation": [
                "Current market indicators suggest prices will increase by {increase}% next month. Locking in today's rate could save you ${savings}.",
                "I can hold this price for {time_period}, but after that, it will increase to ${new_price}/ton. Shall we secure your order now?",
                "We have a limited allocation at this price point. Once we hit {threshold} tons, prices will adjust upward. Would you like to proceed?"
            ],
            "value_proposition": [
                "Our {product} typically helps customers achieve {benefit}% improvement in {metric}. What would that mean for your bottom line?",
                "By switching to our {grade} grade {product}, clients usually see {savings}% reduction in {cost_center}. How would that affect your operations?",
                "With our quality control process, you get {percentage}% less waste than industry standard. At your volume, that's ${savings} in savings."
            ],
            "technical_specs": [
                "Our {product} meets {standard} specifications with {purity}% purity. This translates to {benefit} in your application.",
                "The {grade} grade offers {feature} properties, ideal for {application}. What specifications are critical for your process?",
                "We maintain tight tolerances of {tolerance} on all shipments, ensuring consistent quality. How would this benefit your operation?"
            ],
            "closing_techniques": [
                "Given the {benefit} and potential {savings}% savings, would you like to start with a {quantity} ton order?",
                "We could begin with {quantity} tons at ${price}/ton, then adjust future orders based on your needs. Does that work for you?",
                "I can have this shipped by {date} if we confirm today. Shall I prepare the paperwork?"
            ]
        }

        # Add industry-specific scenarios
        self.industry_scenarios = {
            "construction": [
                "We need steel for a high-rise project",
                "Looking for reinforcement steel",
                "Need structural grade materials"
            ],
            "manufacturing": [
                "Sourcing materials for auto parts",
                "Need metal for machinery components",
                "Looking for industrial grade metals"
            ],
            "electronics": [
                "Need high-purity copper",
                "Sourcing conductive materials",
                "Looking for electronics-grade metals"
            ]
        }

        # Add quality specifications
        self.quality_specs = {
            "copper": {
                "grades": ["Grade A", "Electrolytic", "High Conductivity"],
                "purity": ["99.9%", "99.99%", "99.999%"],
                "standards": ["ASTM B115", "EN 1978", "ISO 1337"]
            },
            "steel": {
                "grades": ["Carbon", "Stainless", "Tool Steel"],
                "types": ["Hot Rolled", "Cold Rolled", "Galvanized"],
                "standards": ["ASTM A36", "AISI 1020", "EN 10025"]
            },
            "aluminum": {
                "grades": ["6061", "7075", "5052"],
                "tempers": ["T6", "T651", "H32"],
                "standards": ["ASTM B209", "EN 573-3", "ISO 6362"]
            }
        }

        # Add delivery and logistics templates
        self.logistics_templates = {
            "shipping_options": [
                "We offer {method} shipping, typically taking {days} days for {quantity} tons",
                "For orders over {threshold} tons, we provide {service_level} shipping at {cost}/ton",
                "We can arrange {transport_type} delivery to {location} within {timeframe}"
            ],
            "delivery_scheduling": [
                "We can deliver {quantity} tons by {date} using {method}",
                "For your location in {city}, delivery time is approximately {days} days",
                "Rush delivery is available, adding ${rush_fee}/ton to expedite to {days} days"
            ]
        }

        # Add complex negotiation scenarios
        self.negotiation_scenarios = {
            "price_objections": [
                {"objection": "Your prices are too high", 
                 "responses": [
                    "While our initial price may be higher, our quality saves you {savings}% in waste reduction. Over a year, that's ${annual_savings}.",
                    "I understand price is important. Let me show you how our {quality_feature} actually reduces your total cost by {savings}%.",
                    "What if I could show you how our premium grade saves you money through reduced reprocessing and higher yields?"
                ]},
                # Add more price objections...
            ],
            "competitor_comparisons": [
                {"situation": "Competitor offering lower price", 
                 "responses": [
                    "I appreciate you sharing that. While they offer ${comp_price}/ton, our certified {grade} grade has {feature} that saves you ${savings}/ton in processing.",
                    "Interesting price point. Have you calculated the total cost including their {weakness} versus our {strength}?",
                    "What aspects of their offer are most appealing? I might be able to structure something similar while maintaining our quality advantage."
                ]},
                # Add more competitor scenarios...
            ]
        }

        # Add industry-specific use cases
        self.industry_use_cases = {
            "automotive": {
                "applications": ["engine components", "body panels", "structural elements"],
                "requirements": ["high tensile strength", "corrosion resistance", "precise tolerances"],
                "certifications": ["ISO/TS 16949", "IATF 16949", "QS-9000"]
            },
            "aerospace": {
                "applications": ["airframe structures", "engine parts", "landing gear"],
                "requirements": ["aerospace grade", "heat resistance", "fatigue strength"],
                "certifications": ["AS9100", "NADCAP", "FAA-PMA"]
            },
            "construction": {
                "applications": ["structural beams", "reinforcement", "cladding"],
                "requirements": ["load bearing", "weather resistance", "code compliance"],
                "certifications": ["ASTM", "EN Standards", "ICC-ES"]
            }
        }

        # Add technical specifications
        self.technical_specs = {
            "metallurgical_properties": {
                "tensile_strength": ["500-550 MPa", "600-650 MPa", "700-750 MPa"],
                "yield_strength": ["250-300 MPa", "350-400 MPa", "450-500 MPa"],
                "elongation": ["20-25%", "25-30%", "30-35%"]
            },
            "quality_assurance": {
                "testing_methods": ["spectrographic analysis", "tensile testing", "hardness testing"],
                "certifications": ["ISO 9001:2015", "ISO 14001", "ISO 45001"],
                "documentation": ["mill test reports", "chemical analysis", "mechanical properties"]
            }
        }

        # Add value proposition scenarios
        self.value_props = {
            "cost_savings": [
                "reduced processing time by {time_saved}%",
                "lower rejection rates by {quality_improvement}%",
                "energy savings of {energy_saved}% in processing"
            ],
            "quality_benefits": [
                "consistent material properties batch-to-batch",
                "certified compliance with {standard}",
                "full traceability from melt to delivery"
            ]
        }

        # Add natural conversation variations
        self.natural_responses = {
            "greetings": [
                "Hey there! What can I help you with today?",
                "Hi! Looking for any metal in particular?",
                "Welcome! How can I assist you with your metal needs?"
            ],
            "price_responses": [
                "Let me check the latest pricing for you... {product} is running at ${price}/ton right now.",
                "I've got {product} at ${price} per ton at the moment. Need any specific quantity?",
                "Current market has {product} at ${price}/ton. Pretty good time to buy, actually.",
                "We're looking at ${price} for {product}. Bulk orders get better rates, by the way."
            ],
            "negotiation_responses": [
                "Look, I get it - price matters. Let me see what I can do...",
                "Between you and me, I might have some wiggle room on larger orders.",
                "Tell you what - if you can commit to {quantity} tons, I'll work something out.",
                "The price is already pretty tight, but maybe we can talk volume discounts?"
            ],
            "technical_casual": [
                "Yeah, this grade is perfect for that. Used it myself in similar projects.",
                "You'll love the quality - our regulars swear by it.",
                "Trust me, this is exactly what you need for that application.",
                "I've got customers using this same grade for {application} with great results."
            ],
            "closing_natural": [
                "Want me to put something together for you?",
                "Should I work up some numbers?",
                "Ready to move forward with this?",
                "Shall we get the paperwork started?"
            ],
            "follow_up": [
                "Just checking in - had a chance to think about it?",
                "Hey, wanted to see if you need any more info?",
                "Circle back when you can - I might have some better numbers for you.",
                "Quick update: prices might be going up next week. Let me know if you want to lock in current rates."
            ]
        }

        # Add conversation flow patterns
        self.conversation_flows = [
            # Natural inquiry flow
            {
                "customer": "Hey, checking prices on copper",
                "agent": "Hey! Yeah, copper's at ${price} right now. What kind of volume are you looking at?",
                "customer": "Maybe around 50 tons",
                "agent": "Nice - that's a good amount. Want me to work up a proper quote for you?"
            },
            # Casual negotiation flow
            {
                "customer": "Bit expensive compared to others",
                "agent": "Yeah, I hear that a lot. But here's the thing - our quality saves you money in the long run. Had a customer just last week tell me they cut waste by 20%.",
                "customer": "Interesting... what kind of quality are we talking about?",
                "agent": "Top-notch stuff. Grade A, certified. Want me to send you some specs?"
            }
        ]

        # Add more natural, casual conversation patterns
        self.casual_responses = {
            "informal_greetings": [
                "Hey! What's up?",
                "Morning! How can I help?",
                "Hi there, what are you looking for today?",
                "Hey, thanks for reaching out!"
            ],
            "casual_price_responses": [
                "Yeah, {product}'s going for ${price} right now. Pretty decent price actually.",
                "Let me check... yep, I can do {product} at ${price}. Need it soon?",
                "I've got {product} at ${price} - that's our best rate this month.",
                "Looking at ${price} for {product}. Between us, might go up next week."
            ],
            "informal_negotiation": [
                "I get it, everyone wants the best deal. Let me see what I can do...",
                "Tell you what - if you can do {quantity} tons, I'll make it worth your while.",
                "Look, I'll level with you - that's already a pretty sharp price, but...",
                "Let me talk to my manager and see if we can work something out."
            ],
            "casual_follow_up": [
                "Just checking in - any thoughts on that quote?",
                "Hey, how's that proposal looking?",
                "Quick update - still interested?",
                "Wanted to loop back about that {product} order."
            ]
        }

        # Add industry-specific casual talk
        self.industry_casual = {
            "construction": [
                "Got a big project coming up?",
                "How's the construction season looking?",
                "Need this for a specific job site?",
                "Running low on site materials?"
            ],
            "manufacturing": [
                "How's production going?",
                "Line running smoothly?",
                "Need to keep the machines fed?",
                "Planning ahead for next quarter?"
            ],
            "automotive": [
                "How's the auto market treating you?",
                "New model in production?",
                "Keeping up with demand?",
                "Supply chain giving you headaches?"
            ]
        }

        # Add Spanish language variations
        self.spanish_responses = {
            "greetings": [
                "¡Hola! ¿En qué puedo ayudarte?",
                "¡Buenos días! ¿Buscas algún metal en particular?",
                "¡Bienvenido! ¿Cómo puedo asistirte hoy?"
            ],
            "price_responses": [
                "El {product} está a ${price} por tonelada en este momento.",
                "Tenemos {product} a ${price} la tonelada. ¿Qué cantidad necesitas?",
                "El precio actual del {product} es ${price}/tonelada. ¿Te interesa?"
            ],
            "negotiation": [
                "Entiendo, el precio es importante. Para pedidos de más de {quantity} toneladas, podemos ofrecer un descuento del {discount}%.",
                "Si puedes comprometerte a {quantity} toneladas, te puedo dar un mejor precio.",
                "Déjame ver qué puedo hacer con el precio para pedidos grandes."
            ]
        }

        # Add Spanish product names and terms
        self.spanish_terms = {
            "products": {
                "copper": "cobre",
                "steel": "acero",
                "aluminum": "aluminio",
                "iron": "hierro",
                "brass": "latón",
                "bronze": "bronce"
            },
            "units": {
                "tons": "toneladas",
                "kilos": "kilos",
                "metric tons": "toneladas métricas"
            }
        }

        # Add regional variations
        self.regional_responses = {
            "us_casual": {
                "greetings": [
                    "Hey! What's up?",
                    "How's it going?",
                    "What can I do ya for?"
                ],
                "closings": [
                    "Let's make it happen!",
                    "Ready to pull the trigger?",
                    "Want me to run the numbers?"
                ]
            },
            "uk_casual": {
                "greetings": [
                    "Hiya! How can I help?",
                    "Morning! Fancy some metal?",
                    "Cheers! What are you after?"
                ],
                "closings": [
                    "Shall we sort this out?",
                    "Fancy moving forward?",
                    "Right then, ready to proceed?"
                ]
            },
            "spanish_regional": {
                "mexico": {
                    "greetings": [
                        "¡Qué tal! ¿En qué te puedo ayudar?",
                        "¡Hola! ¿Qué material necesitas?",
                        "¡Buen día! ¿Buscas algún metal en específico?"
                    ],
                    "closings": [
                        "¿Hacemos el pedido?",
                        "¿Te preparo la cotización?",
                        "¿Procedemos con la orden?"
                    ]
                },
                "spain": {
                    "greetings": [
                        "¡Buenas! ¿Qué tal?",
                        "¡Hola! ¿Qué necesitas?",
                        "¡Buenos días! ¿En qué puedo ayudarte?"
                    ],
                    "closings": [
                        "¿Preparamos el pedido?",
                        "¿Hacemos números?",
                        "¿Formalizamos la compra?"
                    ]
                },
                "puerto_rico": {
                    "greetings": [
                        "¡Wepa! ¿En qué te puedo ayudar?",
                        "¡Saludos! ¿Qué metal estás buscando, pana?",
                        "¡Bendiciones! ¿Qué necesitas hoy?",
                        "¡Qué tal, boricua! ¿En qué te sirvo?"
                    ],
                    "closings": [
                        "¿Bregar con el pedido ahora?",
                        "¿Te monto los números?",
                        "¿Hacemos el deal?",
                        "¿Te preparo el estimate?"
                    ],
                    "casual_responses": [
                        "Mira, ese precio está brutal",
                        "Te puedo dar un deal bien bueno",
                        "Ese metal está volando, pana",
                        "Tremenda calidad, de verdad"
                    ],
                    "negotiation": [
                        "Dale, podemos bregar con eso",
                        "Mira, si coges más cantidad te puedo dar mejor precio",
                        "Ese precio está al bate, pero déjame ver qué puedo hacer",
                        "Tremendo deal te puedo dar en esa cantidad"
                    ],
                    "follow_up": [
                        "¿Qué me dices del estimate que te envié?",
                        "Oye, ¿pudiste revisar los números?",
                        "¿Cómo vamos con eso, pana?",
                        "¿Qué tal si lo bregamos ahora?"
                    ]
                }
            }
        }

        # Add bilingual industry conversations
        self.industry_bilingual = {
            "construction": {
                "english": [
                    {"customer": "Need steel for a skyscraper project", 
                     "agent": "Perfect timing! Our structural steel is ideal for high-rise buildings. How many floors are we talking about?"},
                    {"customer": "Looking for reinforcement bars", 
                     "agent": "Got plenty in stock. What diameter are you looking for?"}
                ],
                "spanish": [
                    {"customer": "Necesito acero para un proyecto de rascacielos",
                     "agent": "¡Perfecto! Nuestro acero estructural es ideal para edificios altos. ¿Cuántos pisos tiene el proyecto?"},
                    {"customer": "Busco varillas de refuerzo",
                     "agent": "Tenemos buen inventario. ¿Qué diámetro necesitas?"}
                ],
                "puerto_rico": [
                    {"customer": "Necesito acero pa' un proyecto en San Juan",
                     "agent": "¡Perfecto! Tenemos todo el material. ¿Cuántos pisos va a tener el edificio, pana?"},
                    {"customer": "Buscando varillas pa' construcción",
                     "agent": "Tenemos tremendo inventario. ¿Qué gauge necesitas?"}
                ]
            },
            "manufacturing": {
                "english": [
                    {"customer": "Need aluminum for auto parts",
                     "agent": "Our automotive grade aluminum is perfect for that. What's your monthly volume?"},
                    {"customer": "Looking for high-grade steel",
                     "agent": "We've got just what you need. What's the application?"}
                ],
                "spanish": [
                    {"customer": "Necesito aluminio para autopartes",
                     "agent": "Nuestro aluminio grado automotriz es perfecto para eso. ¿Cuál es tu volumen mensual?"},
                    {"customer": "Busco acero de alta calidad",
                     "agent": "Tenemos exactamente lo que necesitas. ¿Para qué aplicación es?"}
                ]
            }
        }

        # Add Spanglish variations common in Puerto Rico
        self.spanglish_responses = {
            "price_inquiry": [
                "El {product} está a ${price} el ton. ¿Cuántos quieres ordernar?",
                "Te puedo dar el {product} a ${price}. ¿Qué size necesitas?",
                "Mira, el {product} está a ${price}. ¿Hacemos el quote?"
            ],
            "technical": [
                "La quality es top. Tenemos el certification y todo.",
                "Este {product} es high grade, bien heavy duty.",
                "Es un material bien reliable, pana. Garantizado."
            ],
            "negotiation": [
                "Te puedo hacer un mejor deal si ordenas más quantity",
                "Ese price está hot. Podemos shippear rápido también",
                "Si hacemos el order hoy, te doy un discount brutal"
            ]
        }

        # Add Puerto Rican industry-specific terminology
        self.puerto_rican_industry = {
            "construction": {
                "terms": {
                    "building": "bildin",
                    "project": "proyecto",
                    "site": "site",
                    "blueprint": "blueprint",
                    "materials": "materiales",
                    "delivery": "delivery"
                },
                "phrases": [
                    "Necesito material pal site en Bayamón",
                    "El contractor necesita el steel pa' mañana",
                    "Tengo un proyecto en el área metro",
                    "El boss quiere saber el precio del material"
                ],
                "responses": [
                    "Dale, te puedo tener eso en el site pa' {date}",
                    "Brutal, tenemos todo el material ready pal proyecto",
                    "Te lo puedo deliver directo al site, pana",
                    "Ese steel es perfect pal proyecto que tienes"
                ]
            },
            "manufacturing": {
                "terms": {
                    "factory": "factoría",
                    "machine": "máquina",
                    "production": "producción",
                    "quality": "calidad",
                    "parts": "piezas"
                },
                "phrases": [
                    "La máquina está down, necesito el material ASAP",
                    "El quality control está pidiendo mejor grade",
                    "Necesito hacer un order pa' la factoría",
                    "El manager quiere ver los specs del material"
                ],
                "responses": [
                    "Tranquilo, te puedo resolver con un material high quality",
                    "Ese grade es perfect pa' tu máquina",
                    "Te puedo tener el material en la factoría mañana mismo",
                    "Los specs son top notch, no vas a tener issues"
                ]
            },
            "common_expressions": [
                "Está cabrón ese precio",
                "Mano, eso está brutal",
                "Dale pa' lante con eso",
                "Eso está al bate",
                "Tremendo deal",
                "Está heavy ese material"
            ],
            "closing_phrases": [
                "¿Qué dices, lo bregamos?",
                "¿Hacemos el deal?",
                "¿Te monto el order?",
                "¿Cerramos con eso?",
                "¿Te hago el quote ahora mismo?"
            ]
        }

    def calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score based on keywords"""
        text = text.lower()
        score = 0
        total_keywords = 0
        
        for category, keywords in self.relevant_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword in text:
                    category_score += 1
            score += category_score / len(keywords)
            total_keywords += 1
        
        return score / total_keywords

    def clean_and_validate_conversation(self, customer_msg: str, agent_msg: str) -> Tuple[str, str, bool]:
        """Clean and validate conversation pairs"""
        try:
            # Basic cleaning
            customer_msg = self.clean_conversation(customer_msg)
            agent_msg = self.clean_conversation(agent_msg)
            
            # Length validation
            if (len(customer_msg) < self.quality_thresholds["min_customer_length"] or
                len(agent_msg) < self.quality_thresholds["min_agent_length"] or
                len(customer_msg) > self.quality_thresholds["max_message_length"] or
                len(agent_msg) > self.quality_thresholds["max_message_length"]):
                return None, None, False
            
            # Calculate relevance scores
            customer_relevance = self.calculate_relevance_score(customer_msg)
            agent_relevance = self.calculate_relevance_score(agent_msg)
            
            # Check if meets quality threshold
            if (customer_relevance + agent_relevance) / 2 < self.quality_thresholds["min_relevance_score"]:
                return None, None, False
            
            # Additional cleaning
            customer_msg = self.enhance_text_quality(customer_msg)
            agent_msg = self.enhance_text_quality(agent_msg)
            
            return customer_msg, agent_msg, True
            
        except Exception as e:
            print(f"Error in conversation validation: {e}")
            return None, None, False

    def enhance_text_quality(self, text: str) -> str:
        """Enhance text quality with additional cleaning"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Standardize product units
        text = re.sub(r'tonnes?', 'tons', text, flags=re.IGNORECASE)
        text = re.sub(r'k(?:ilo)?g(?:ram)?s?', 'kg', text, flags=re.IGNORECASE)
        
        # Format prices consistently
        text = re.sub(r'\$\s*(\d+)', r'$\1', text)
        text = re.sub(r'(\d+)\s*dollars', r'$\1', text, flags=re.IGNORECASE)
        
        # Fix common typos
        text = text.replace('aluminium', 'aluminum')
        text = text.replace('tonnes', 'tons')
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.,!?])(\w)', r'\1 \2', text)
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()

    async def extract_conversations_from_html(self, html_content: str) -> List[Dict]:
        """Extract conversations from HTML content with better parsing"""
        soup = BeautifulSoup(html_content, 'html.parser')
        conversations = []
        
        # Remove irrelevant elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # Get main content
        content = soup.get_text()
        
        # Extract conversations using patterns
        for pattern in self.conversation_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    customer_msg, agent_msg, is_valid = self.clean_and_validate_conversation(
                        match.group(1),
                        match.group(2)
                    )
                    
                    if is_valid:
                        conversations.append({
                            "customer_message": customer_msg,
                            "agent_response": agent_msg,
                            "quality_score": self.calculate_relevance_score(customer_msg + " " + agent_msg),
                            "source": "web_scrape",
                            "extraction_date": datetime.now().isoformat()
                        })
                except Exception:
                    continue
        
        return conversations

    async def generate_variations(self, session: AsyncSession):
        """Generate large number of training variations"""
        variations = set()  # Use a set to prevent duplicates
        max_variations = 1000  # Set a reasonable limit
        
        # Generate more variations per product
        for product in self.variations["products"]:
            if len(variations) >= max_variations:
                break
                
            base_price = random.randint(*self.product_prices[product])
            
            # Price inquiries (max 20 variations per product)
            for template in self.variations["price_inquiry"]:
                for _ in range(20):  # Increase number of variations
                    if len(variations) >= max_variations:
                        break
                        
                    price = base_price + random.randint(-100, 100)
                    customer_msg = template.format(product=product)
                    agent_response = random.choice(self.response_templates["price_inquiry"]).format(
                        product=product,
                        price=price,
                        min_order=self._get_min_order(product)
                    )
                    variation = self._create_variation("price_inquiry", customer_msg, agent_response, product, price)
                    # Convert dict to tuple for set storage
                    variation_tuple = (variation["customer_message"], variation["agent_response"])
                    variations.add(variation_tuple)
        
        # Store unique variations
        print(f"Generating {len(variations)} unique variations...")
        stored_count = 0
        for customer_msg, agent_response in variations:
            try:
                await store_training_example(
                    session,
                    customer_message=customer_msg,
                    agent_response=agent_response,
                    metadata={
                        "intent": "price_inquiry",
                        "variation_type": "generated",
                        "generation_time": datetime.now().isoformat()
                    }
                )
                stored_count += 1
            except Exception as e:
                print(f"Error storing variation: {e}")
                continue
        
        return stored_count

    def _create_variation(self, intent: str, customer_msg: str, agent_response: str, product: str, base_price: int) -> dict:
        """Helper to create variation with metadata"""
        return {
            "customer_message": customer_msg,
            "agent_response": agent_response,
            "metadata": {
                "intent": intent,
                "product": product,
                "base_price": base_price,
                "variation_type": "generated",
                "generation_time": datetime.now().isoformat()
            }
        }

    async def get_suitecrm_samples(self) -> List[Dict]:
        """Collect example conversations from SuiteCRM demo data"""
        # This was the missing method
        return []  # For now, return empty list as we focus on variations

    async def gather_training_data(self, session: AsyncSession):
        """Gather training data from various sources"""
        try:
            conversations = []
            
            # Get HuggingFace dataset examples
            huggingface_count = await import_huggingface_datasets(session)
            
            # Generate variations
            variation_count = await self.generate_variations(session)
            
            # Get examples from other sources
            conversations.extend(await self.get_sales_hacker_data())
            conversations.extend(await self.get_hubspot_examples())
            
            # Store collected conversations
            stored_count = 0
            for conv in conversations:
                try:
                    await store_training_example(session, **conv)
                    stored_count += 1
                except Exception as e:
                    print(f"Error storing conversation: {e}")
                    continue
            
            total_count = huggingface_count + variation_count + stored_count
            print(f"Total examples collected: {total_count}")
            return total_count
            
        except Exception as e:
            print(f"Error gathering training data: {e}")
            return 0

    async def get_sales_hacker_data(self) -> List[Dict]:
        """Collect sales conversation examples from Sales Hacker"""
        conversations = []
        async with aiohttp.ClientSession() as session:
            try:
                # Get article URLs
                async with session.get("https://www.saleshacker.com/tag/sales-conversations/") as response:
                    soup = BeautifulSoup(await response.text(), 'html.parser')
                    articles = soup.find_all('article')
                    
                    for article in articles:
                        try:
                            article_url = article.find('a')['href']
                            async with session.get(article_url) as article_response:
                                content = await article_response.text()
                                convs = self.extract_conversations(content)
                                for customer_msg, agent_msg in convs:
                                    conversations.append({
                                        "customer_message": customer_msg,
                                        "agent_response": agent_msg,
                                        "source": "sales_hacker",
                                        "category": "best_practices",
                                        "success_metrics": {"published_example": True}
                                    })
                        except Exception as e:
                            print(f"Error processing article: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error fetching Sales Hacker data: {e}")
        
        return conversations

    async def get_hubspot_examples(self) -> List[Dict]:
        """Collect sales conversation examples from HubSpot's blog"""
        conversations = []
        async with aiohttp.ClientSession() as session:
            try:
                urls = [
                    "https://blog.hubspot.com/sales/sales-conversation-starters",
                    "https://blog.hubspot.com/sales/sales-scripts-examples"
                ]
                
                for url in urls:
                    async with session.get(url) as response:
                        content = await response.text()
                        convs = self.extract_conversations(content)
                        for customer_msg, agent_msg in convs:
                            conversations.append({
                                "customer_message": customer_msg,
                                "agent_response": agent_msg,
                                "source": "hubspot",
                                "category": "sales_scripts",
                                "success_metrics": {"published_example": True}
                            })
                            
            except Exception as e:
                print(f"Error fetching HubSpot data: {e}")
        
        return conversations

    def extract_conversations(self, content: str) -> List[Tuple[str, str]]:
        """Extract conversation pairs from text content"""
        conversations = []
        
        for pattern in self.conversation_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    customer_msg = match.group(1).strip()
                    agent_msg = match.group(2).strip()
                    if len(customer_msg) > 10 and len(agent_msg) > 10:
                        conversations.append((customer_msg, agent_msg))
                except Exception:
                    continue
        
        return conversations

    def clean_conversation(self, text: str) -> str:
        """Clean and normalize conversation text"""
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?$%()-]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    async def generate_advanced_variations(self, session: AsyncSession):
        """Generate more complex variations"""
        variations = []
        
        # Generate technical inquiry variations
        for product in self.variations["products"]:
            base_price = random.randint(*self.product_prices[product])
            specs = self.technical_specs["metallurgical_properties"]
            for prop, values in specs.items():
                customer_msg = f"What's the {prop} of your {product}?"
                value = random.choice(values)
                agent_response = f"Our {product} has a {prop} of {value}, exceeding industry standards. This ensures optimal performance in your application. Would you like to discuss quantities?"
                variations.append(self._create_variation("technical", customer_msg, agent_response, product, base_price))

        # Generate industry-specific variations
        for industry, details in self.industry_use_cases.items():
            for app in details["applications"]:
                for product in self.variations["products"]:
                    customer_msg = f"We need {product} for {app} in {industry}"
                    cert = random.choice(details["certifications"])
                    req = random.choice(details["requirements"])
                    agent_response = f"Perfect! Our {product} is {cert} certified and specifically designed for {app}, ensuring {req}. What quantity do you need?"
                    variations.append(self._create_variation("industry_specific", customer_msg, agent_response, product))

        # Generate value proposition variations
        for product in self.variations["products"]:
            for value_prop in self.value_props["cost_savings"]:
                savings = random.randint(10, 30)
                customer_msg = f"How can {product} improve our efficiency?"
                agent_response = value_prop.format(
                    time_saved=savings,
                    quality_improvement=savings+5,
                    energy_saved=savings-3
                )
                variations.append(self._create_variation("value_prop", customer_msg, agent_response, product))

        # Store all variations
        print(f"Generating {len(variations)} advanced variations...")
        for var in variations:
            await store_training_example(session, **var)
        
        return len(variations)

    async def generate_natural_variations(self, session: AsyncSession):
        """Generate more natural-sounding conversations"""
        variations = []
        
        for product in self.variations["products"]:
            base_price = random.randint(*self.product_prices[product])
            
            # Mix and match different response styles
            for _ in range(10):  # Generate 10 variations per product
                greeting = random.choice(self.natural_responses["greetings"])
                price_response = random.choice(self.natural_responses["price_responses"])
                closing = random.choice(self.natural_responses["closing_natural"])
                
                # Combine them naturally with some randomization
                agent_response = f"{greeting} {price_response.format(product=product, price=base_price)}"
                if random.random() > 0.5:  # 50% chance to add closing
                    agent_response += f" {closing}"
                
                variations.append({
                    "customer_message": f"Hi, what's the price of {product}?",
                    "agent_response": agent_response,
                    "metadata": {
                        "intent": "price_inquiry",
                        "product": product,
                        "style": "natural",
                        "base_price": base_price
                    }
                })
        
        # Store variations
        print(f"Generating {len(variations)} natural variations...")
        for var in variations:
            await store_training_example(session, **var)
        
        return len(variations)

    async def generate_bilingual_variations(self, session: AsyncSession):
        """Generate variations in both English and Spanish"""
        variations = []
        
        for product in self.variations["products"]:
            base_price = random.randint(*self.product_prices[product])
            spanish_product = self.spanish_terms["products"][product]
            
            # Generate price inquiries in both languages
            for _ in range(5):  # 5 variations per product per language
                # English variation
                eng_greeting = random.choice(self.natural_responses["greetings"])
                eng_price = random.choice(self.natural_responses["price_responses"])
                eng_response = f"{eng_greeting} {eng_price.format(product=product, price=base_price)}"
                
                # Spanish variation
                esp_greeting = random.choice(self.spanish_responses["greetings"])
                esp_price = random.choice(self.spanish_responses["price_responses"])
                esp_response = f"{esp_greeting} {esp_price.format(product=spanish_product, price=base_price)}"
                
                variations.extend([
                    {
                        "customer_message": f"What's the price of {product}?",
                        "agent_response": eng_response,
                        "metadata": {"language": "english", "product": product, "style": "natural"}
                    },
                    {
                        "customer_message": f"¿Cuál es el precio del {spanish_product}?",
                        "agent_response": esp_response,
                        "metadata": {"language": "spanish", "product": product, "style": "natural"}
                    }
                ])
        
        # Store variations
        print(f"Generating {len(variations)} bilingual variations...")
        for var in variations:
            await store_training_example(session, **var)
        
        return len(variations)

    async def generate_puerto_rican_variations(self, session: AsyncSession):
        """Generate variations with Puerto Rican Spanish"""
        variations = []
        
        for product in self.variations["products"]:
            base_price = random.randint(*self.product_prices[product])
            spanish_product = self.spanish_terms["products"][product]
            
            # Generate variations for each industry
            for industry in ["construction", "manufacturing"]:
                industry_terms = self.puerto_rican_industry[industry]["terms"]
                industry_phrases = self.puerto_rican_industry[industry]["phrases"]
                industry_responses = self.puerto_rican_industry[industry]["responses"]
                
                for phrase in industry_phrases:
                    # Create natural conversation flow
                    greeting = random.choice(self.regional_responses["spanish_regional"]["puerto_rico"]["greetings"])
                    response = random.choice(industry_responses)
                    closing = random.choice(self.puerto_rican_industry["closing_phrases"])
                    expression = random.choice(self.puerto_rican_industry["common_expressions"])
                    
                    # Build response with natural flow and Spanglish
                    agent_response = f"{greeting} {response.format(date='next week')}. {expression}. {closing}"
                    
                    variations.append({
                        "customer_message": phrase,
                        "agent_response": agent_response,
                        "metadata": {
                            "language": "spanish",
                            "dialect": "puerto_rican",
                            "industry": industry,
                            "product": product,
                            "style": "natural"
                        }
                    })
        
        # Store variations
        print(f"Generating {len(variations)} Puerto Rican industry variations...")
        for var in variations:
            await store_training_example(session, **var)
        
        return len(variations)

    def _get_min_order(self, product: str) -> int:
        """Get minimum order quantity for a product"""
        min_orders = {
            "steel": 100,
            "copper": 50,
            "aluminum": 75,
            "zinc": 100,
            "nickel": 25
        }
        return min_orders.get(product.lower(), 50)  # Default 50 tons 