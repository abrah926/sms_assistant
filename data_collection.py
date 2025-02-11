from models import TrainingExample, Message, MessageType
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.sql import text
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import List, Dict
import csv

async def store_training_example(
    session: AsyncSession,
    customer_message: str,
    agent_response: str,
    context: str = None,
    metadata: dict = None
):
    """Store a new training example"""
    example = TrainingExample(
        customer_message=customer_message,
        agent_response=agent_response,
        context=context,
        metadata=metadata or {}
    )
    session.add(example)
    await session.commit()
    return example

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
    
    # Query to find successful conversations (e.g., ones that led to orders or positive responses)
    query = text("""
        WITH conversation_pairs AS (
            SELECT 
                m1.phone,
                m1.content as customer_message,
                m2.content as agent_response,
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
                -- Filter for successful interactions
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
        if source_type == "hubspot":
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