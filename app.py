import warnings
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, validator
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from models import Message, Base, MessageType
from config import DATABASE_URL, OPENAI_API_KEY, ENABLE_METRICS
from utils import (
    sanitize_phone_number, sanitize_message,
    classify_message_type, MessageMetadata
)
from monitoring import MetricsMiddleware, message_counter, response_time, error_counter
import openai
from fastapi.responses import Response, JSONResponse
from cachetools import TTLCache
from contextlib import asynccontextmanager
import re
from llm import MistralLLM
from fastapi.middleware.cors import CORSMiddleware

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

@app.on_event("startup")
async def startup_event():
    print("Starting application...")
    try:
        # Initialize database
        print("Initializing database...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database initialized")
        
        # Initialize OpenAI
        print("Setting OpenAI key...")
        openai.api_key = OPENAI_API_KEY
        
        # Initialize LLM
        print("Initializing LLM...")
        global llm
        llm = MistralLLM()
        print("LLM initialized")
        
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
    """Get recent message history for context"""
    query = select(Message).where(Message.phone == phone).order_by(Message.timestamp.desc()).limit(limit)
    result = await db.execute(query)
    messages = result.scalars().all()
    return messages

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
    """Generate AI response using Mistral"""
    try:
        response = await llm.generate(
            prompt=message,
            history=history,
            db=db
        )
        return await clean_ai_response(response)
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I received your message. How can I help you today?"

@app.post("/message/webhook")
@response_time.time()
async def handle_message(
    message: SMSMessage,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    # Add method validation
    if request.method != "POST":
        raise HTTPException(
            status_code=405,
            detail="Method not allowed. Use POST."
        )

    try:
        # Process and classify message
        message_type = classify_message_type(message.body)
        metadata = MessageMetadata(message.body)
        
        print(f"Processing message: {message.body}")
        print(f"Message type: {message_type}")
        
        # Store incoming message
        new_message = Message(
            phone=message.from_number,
            content=message.body,
            direction="incoming",
            timestamp=datetime.now(timezone.utc),
            message_type=message_type,
            meta_data=metadata.to_json()
        )
        
        # Use add() instead of vars()
        db.add(new_message)
        await db.commit()
        
        message_counter.labels(direction='incoming', type=message_type.value).inc()
        
        # Get message history for context
        history = await get_message_history(message.from_number, db)
        print(f"Got message history: {len(history)} messages")
        
        # Generate AI response with context
        try:
            # Make sure to await the response
            response = await generate_ai_response(message.body, history, message_type, db)
            print(f"Generated response: {response}")
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise
        
        # Store outgoing message
        out_message = Message(
            phone=message.from_number,
            content=response,
            direction="outgoing",
            timestamp=datetime.now(timezone.utc),
            message_type=message_type
        )
        db.add(out_message)
        await db.commit()
        
        message_counter.labels(direction='outgoing', type=message_type.value).inc()
        
        return {"status": "success", "response": response, "type": message_type.value}
        
    except Exception as e:
        print(f"Error in handle_message: {str(e)}")
        print(f"Error type: {type(e)}")
        await db.rollback()
        error_counter.labels(type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

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