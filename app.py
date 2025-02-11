import warnings
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, validator
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, create_engine
from models import Message, Base, MessageType
from config import DATABASE_URL, OPENAI_API_KEY, ENABLE_METRICS
from utils import (
    sanitize_phone_number, sanitize_message,
    classify_message_type, MessageMetadata,
    get_fallback_response
)
from business_functions import get_customer_context
from monitoring import MetricsMiddleware, message_counter, response_time, error_counter
import openai
from fastapi.responses import Response, JSONResponse
from cachetools import TTLCache
from contextlib import asynccontextmanager
import re
from llm import MistralLLM
from fastapi.middleware.cors import CORSMiddleware
from queue_handler import MessageQueue
import asyncio

# Database setup
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Rate limiting cache
rate_limit_cache = TTLCache(maxsize=100, ttl=60)  # 60 seconds TTL

class SMSMessage(BaseModel):
    from_number: str
    body: str
    
    @validator('from_number')
    def validate_phone(cls, v):
        clean_number = sanitize_phone_number(v)
        if not clean_number:
            raise ValueError('Invalid phone number')
        return clean_number
    
    @validator('body')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return sanitize_message(v)

# Create app first
app = FastAPI()

# Add middlewares
if ENABLE_METRICS:
    app.add_middleware(MetricsMiddleware)  # No arguments needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance
llm = None

# Global queue handler
message_queue = None

@app.on_event("startup")
async def startup_event():
    print("Starting application...")
    try:
        # Initialize database
        print("Initializing database...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database initialized")
        
        # Initialize LLM
        print("Initializing LLM...")
        global llm
        llm = MistralLLM()
        print("LLM initialized")
        
        # Schedule training data collection
        async def collect_training_data():
            while True:
                async with AsyncSessionLocal() as session:
                    await schedule_training_collection(session)
                await asyncio.sleep(3600)  # Run every hour
        
        asyncio.create_task(collect_training_data())
        
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    await engine.dispose()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def get_message_history(phone: str, db: AsyncSession, limit: int = 5):
    query = select(Message).where(Message.phone == phone).order_by(Message.timestamp.desc()).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())  # Convert to list immediately

async def clean_ai_response(response: str) -> str:
    """Clean and humanize AI response"""
    # Remove references and citations
    response = re.sub(r'\[\d+\]|\[citation needed\]', '', response)
    
    # Remove special characters but keep basic punctuation
    response = re.sub(r'[^a-zA-Z0-9\s.,!?¿¡$%:;()-]', '', response)
    
    # Remove multiple spaces
    response = ' '.join(response.split())
    
    # Remove AI-like prefixes
    prefixes = ['AI:', 'Assistant:', 'Bot:', 'Response:']
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    return response.strip()

async def generate_ai_response(message: str, history: list[Message], message_type: MessageType, db: AsyncSession) -> str:
    """Simple wrapper around LLM generate"""
    try:
        if not llm:
            return "System is initializing. Please try again in a moment."
        return await llm.generate(message, history, db)
    except Exception as e:
        print(f"Error in generate_ai_response: {e}")
        return "I received your message. How can I help you today?"

def process_message_sync(phone: str, content: str, db_url: str):
    """Synchronous message processing"""
    try:
        sync_url = db_url.replace('+asyncpg', '')
        engine = create_engine(sync_url)
        Session = sessionmaker(engine)
        
        with Session() as session:
            try:
                # Check for recent duplicate messages
                recent_message = session.query(Message)\
                    .filter(
                        Message.phone == phone,
                        Message.content == content,
                        Message.direction == "incoming",
                        Message.timestamp >= datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).first()
                
                if recent_message:
                    print("Duplicate message detected, skipping")
                    return None

                # Store incoming message
                message = Message(
                    phone=phone,
                    content=content,
                    direction="incoming",
                    timestamp=datetime.now(timezone.utc),
                    message_type=MessageType.GENERAL,
                    meta_data={"processed": False}
                )
                session.add(message)
                session.flush()  # Flush before generating response

                # Use synchronous response generation
                response = llm.generate_sync(content, [], None)
                
                # Store response
                out_message = Message(
                    phone=phone,
                    content=response,
                    direction="outgoing",
                    timestamp=datetime.now(timezone.utc),
                    message_type=MessageType.GENERAL,
                    meta_data={"processed": True}
                )
                session.add(out_message)
                session.commit()  # Single commit for both messages
                return response

            except Exception as inner_e:
                session.rollback()
                raise
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise

@app.get("/")
async def read_root():
    return {"message": "Welcome to the SMS Agent API"}

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={"message": "Favicon not set"})

@app.post("/message/webhook")
async def message_webhook(
    request: Request, 
    background_tasks: BackgroundTasks
):
    try:
        print("Received webhook request")
        data = await request.json()
        print(f"Request data: {data}")
        
        if 'from_number' not in data or 'body' not in data:
            print("Missing required fields")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing required fields: from_number and body"
                }
            )
        
        async with AsyncSessionLocal() as session:
            print(f"\n=== Processing new message ===")
            print(f"From: {data['from_number']}")
            print(f"Message: {data['body']}")
            
            # Fetch customer context
            context = await get_customer_context(data['from_number'], session)
            print(f"\nContext retrieved: {context}")
            
            # Generate response with context
            print("\nGenerating response...")
            response = llm.generate_sync(data['body'], context, session)
            print(f"Generated response: {response}")
            
            # Store messages
            print("\nStoring messages...")
            message = Message(
                phone=data['from_number'],
                content=data['body'],
                direction="incoming",
                timestamp=datetime.now(timezone.utc),
                message_type=MessageType.GENERAL,
                meta_data={"processed": False}
            )
            session.add(message)
            
            out_message = Message(
                phone=data['from_number'],
                content=response,
                direction="outgoing",
                timestamp=datetime.now(timezone.utc),
                message_type=MessageType.GENERAL,
                meta_data={"processed": True}
            )
            session.add(out_message)
            
            print("Committing to database...")
            await session.commit()
            print("Database commit successful")
            
            return JSONResponse(content={
                "status": "success",
                "message": "Request processed",
                "response": response,
                "received": {
                    "from": data['from_number'],
                    "body_length": len(data['body'])
                }
            })
            
    except Exception as e:
        print(f"\n=== Error in webhook ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Error details: {getattr(e, 'detail', 'No details available')}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "type": str(type(e))
            }
        )

@app.get("/message/webhook")
async def message_webhook_get():
    """Handle GET requests with helpful message"""
    return JSONResponse(
        status_code=405,
        content={
            "status": "error",
            "message": "This endpoint requires a POST request. Example usage:",
            "example": {
                "curl": 'curl -X POST http://localhost:8000/message/webhook -H "Content-Type: application/json" -d \'{"from_number": "+1234567890", "body": "What is the price of copper?"}\''
            }
        }
    )

@app.get("/metrics")
async def get_metrics():
    from prometheus_client import generate_latest
    return Response(content=generate_latest())

async def rate_limiter(request: Request):
    client_ip = request.client.host
    now = datetime.now(timezone.utc)
    
    if client_ip in rate_limit_cache:
        if rate_limit_cache[client_ip] >= 60:  # 60 requests per minute
            raise HTTPException(status_code=429, detail="Too many requests")
        rate_limit_cache[client_ip] += 1
    else:
        rate_limit_cache[client_ip] = 1

@app.get("/health")
async def health_check():
    try:
        # Check database connection
        async with AsyncSession(engine) as session:
            try:
                result = await session.execute(select(1))
                row = result.scalar()  # Use scalar() instead of first() or scalar_one()
                if row != 1:
                    raise Exception("Database check failed")
            except Exception as db_error:
                print(f"Database check failed: {str(db_error)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "component": "database",
                        "detail": str(db_error)
                    }
                )
        
        # Check LLM initialization
        try:
            if llm is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "component": "llm",
                        "detail": "LLM not initialized"
                    }
                )
            # Test LLM is working
            tokenizer = llm.tokenizer  # Just check if we can access the tokenizer
        except Exception as llm_error:
            print(f"LLM check failed: {str(llm_error)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "component": "llm",
                    "detail": str(llm_error)
                }
            )
        
        return {
            "status": "ok",
            "database": "connected",
            "llm": "initialized"
        }
    except Exception as e:
        print(f"Health check error: {str(e)}")
        print(f"Error type: {type(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "component": "general",
                "detail": str(e)
            }
        )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"Error details: {str(exc)}")  # Debug print
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": getattr(exc, "detail", str(exc))
        }
    )

@app.post("/test/webhook")
async def test_webhook(request: Request):
    try:
        data = await request.json()
        return JSONResponse(content={
            "status": "success",
            "response": "This is a test response",
            "echo": data
        })
    except Exception as e:
        print(f"Test Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )