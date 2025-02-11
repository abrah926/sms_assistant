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

    async def gather_training_data(self, session: AsyncSession):
        """Main function to gather and store training data"""
        try:
            # Collect from multiple sources
            conversations = []
            conversations.extend(await self.get_sales_hacker_data())
            conversations.extend(await self.get_hubspot_examples())
            conversations.extend(await self.get_suitecrm_samples())
            
            # Store collected conversations
            stored_count = 0
            for conv in conversations:
                try:
                    await store_training_example(
                        session,
                        conv["customer_message"],
                        conv["agent_response"],
                        metadata={
                            "source": conv["source"],
                            "category": conv["category"],
                            "success_metrics": conv.get("success_metrics", {}),
                            "industry": conv.get("industry"),
                            "product_type": conv.get("product_type")
                        }
                    )
                    stored_count += 1
                except Exception as e:
                    print(f"Error storing conversation: {e}")
                    continue
            
            return stored_count
            
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

# Add to INITIAL_TRAINING_DATA
SALES_TECHNIQUE_DATA = [
    # SPIN Selling examples
    {
        "customer_message": "We're having trouble with our current metal supplier's delivery times",
        "agent_response": "That must be impacting your production schedule. How many delays have you experienced in the last month?",
        "metadata": {"technique": "spin_selling", "stage": "problem"}
    },
    # Challenger Sale examples
    {
        "customer_message": "Your prices are higher than other suppliers",
        "agent_response": "Did you know that our quality control process reduces waste by 15%? Most companies actually save money with us despite the higher initial cost. What's your current waste percentage?",
        "metadata": {"technique": "challenger_sale", "stage": "reframe"}
    }
] 