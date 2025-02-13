import warnings
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, validator
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, create_engine, func
from models import Message, Base, MessageType, TrainingExample
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
from data_collection import (
    SalesDataCollector, 
    collect_successful_conversations,
    store_training_example
)
from training import SalesModelTrainer
import json
from typing import Dict
from threading import Lock
import copy

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

# Add with other global variables
llm = None
message_queue = None
collector = None

# Initialize training status
training_status = {
    "active": False,
    "progress": 0,
    "total": 50,
    "accuracy": 0.0,
    "status": "not started"
}

# Add at the top with other globals
training_status_lock = Lock()

def update_training_status(updates: dict):
    """Thread-safe update of training status"""
    global training_status
    with training_status_lock:
        training_status.update(updates)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting application...")
    try:
        # Initialize database
        print("Initializing database...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database initialized")
        
        # Initialize LLM and collector
        print("Initializing services...")
        global llm, collector
        llm = MistralLLM()
        collector = SalesDataCollector()
        
        # Start background data collection
        asyncio.create_task(collect_training_data())
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    await engine.dispose()

# Then create the FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

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
    """Test endpoint"""
    print("DEBUG: Root endpoint accessed")  # Debug print
    return {"status": "ok", "message": "API is running"}

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

@app.get("/training/count")
async def get_training_count() -> Dict:
    """Get current count of training examples and status"""
    try:
        print("DEBUG: Accessing training count endpoint")
        
        # Add timeout to prevent hanging
        async def get_status():
            return training_status.copy()
            
        try:
            # Wait max 2 seconds for status
            current_status = await asyncio.wait_for(get_status(), timeout=2.0)
        except asyncio.TimeoutError:
            print("DEBUG: Status retrieval timed out")
            return {
                "count": 0,
                "total": 50,
                "accuracy": "0.00%",
                "status": "timeout",
                "message": "Status retrieval timed out, training may be busy"
            }
        
        print(f"DEBUG: Current status: {current_status}")
        
        # Add more detailed status info
        response = {
            "count": current_status.get("progress", 0),
            "total": current_status.get("total", 50),
            "accuracy": f"{current_status.get('accuracy', 0)*100:.2f}%",
            "status": current_status.get("status", "not started"),
            "active": current_status.get("active", False),
            "last_update": datetime.now().isoformat()
        }
        
        print(f"DEBUG: Sending response: {response}")
        return response
        
    except Exception as e:
        print(f"DEBUG: Error in training count: {str(e)}")
        return {
            "count": 0,
            "total": 50,
            "accuracy": "0.00%",
            "status": "error",
            "error": str(e)
        }

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Start training in background"""
    global training_status  # Make sure we're modifying the global status
    
    if training_status["active"]:
        return JSONResponse(content={
            "status": "error",
            "message": "Training already in progress"
        })

    training_status.update({
        "active": True,
        "progress": 0,
        "total": 50,
        "accuracy": 0.0,
        "status": "training"
    })
    
    background_tasks.add_task(start_training)
    return {"message": "Training started in background"}

async def start_training():
    try:
        print("\n=== Starting model training ===")
        async with AsyncSessionLocal() as session:
            # First get total examples
            query = select(func.count(TrainingExample.id))
            result = await session.execute(query)
            total_examples = result.scalar()
            print(f"Training with {total_examples} examples")
            
            trainer = SalesModelTrainer(llm.model, llm.tokenizer)
            
            def update_status(iteration: int, accuracy: float):
                update_training_status({
                    "progress": iteration,
                    "accuracy": accuracy,
                    "status": "training"
                })
                print(f"Iteration {iteration}/50 - Accuracy: {accuracy*100:.2f}%")
            
            trained_model = await trainer.train_iteratively(
                session,
                iterations=50,
                status_callback=update_status
            )
            
            if trained_model:
                print("Training complete. Updating model...")
                llm.model = trained_model
                llm.model.eval()
                
                update_training_status({
                    "active": False,
                    "status": "completed",
                    "progress": 50,
                    "accuracy": 0.95
                })
                
                print("=== Training completed successfully ===\n")
                
    except Exception as e:
        print(f"Training error: {str(e)}")
        update_training_status({
            "active": False,
            "progress": 0,
            "status": "error",
            "error_message": str(e)
        })

@app.get("/training/examples")
async def get_training_examples():
    """Get sample of training examples"""
    async with AsyncSessionLocal() as session:
        query = select(TrainingExample).order_by(TrainingExample.created_at.desc()).limit(5)
        result = await session.execute(query)
        examples = result.scalars().all()
        
        return {
            "examples": [
                {
                    "customer_message": ex.customer_message,
                    "agent_response": ex.agent_response,
                    "metadata": ex.meta_info
                }
                for ex in examples
            ]
        }

@app.get("/training/ab-results")
async def get_ab_results():
    """Get A/B test results for review"""
    try:
        with open('ab_test_results.json', 'r') as f:
            results = json.load(f)
            
        return JSONResponse(content={
            "status": "success",
            "total_pairs": results['total_examples'],
            "sample_pairs": results['ab_pairs'][:10],  # Show first 10 pairs
            "message": "Use /training/ab-results/select to choose preferred responses"
        })
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No A/B test results available"}
        )

@app.post("/training/ab-results/select")
async def select_ab_responses(request: Request):
    """Select preferred responses from A/B tests"""
    data = await request.json()
    selections = data.get('selections', [])
    
    async with AsyncSessionLocal() as session:
        for selection in selections:
            message_id = selection['message_id']
            preferred_version = selection['preferred']  # 'a' or 'b'
            
            # Store the preferred response as training example
            await store_training_example(
                session,
                customer_message=selection['customer_message'],
                agent_response=selection[f'variation_{preferred_version}']['response'],
                metadata={"source": "ab_test_selected", "version": preferred_version}
            )
    
    return JSONResponse(content={
        "status": "success",
        "message": f"Stored {len(selections)} preferred responses"
    })

@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return JSONResponse(content={
        "active": training_status["active"],
        "progress": training_status["progress"],
        "total_iterations": training_status["total"],
        "percentage": f"{(training_status['progress'] / training_status['total']) * 100:.1f}%"
    })

async def collect_training_data():
    """Periodic data collection task"""
    while True:
        async with AsyncSessionLocal() as session:
            print("\n=== Starting data collection ===")
            # Collect from external sources
            count = await collector.gather_training_data(session)
            print(f"Collected {count} new examples from external sources")
            
            # Collect internal conversations
            internal_count = await collect_successful_conversations(session)
            print(f"Collected {internal_count} internal examples")
            
            print("=== Data collection complete ===\n")
            
        await asyncio.sleep(86400)  # Run daily

@app.get("/debug/training")
async def debug_training():
    """Debug endpoint to check training state"""
    try:
        # Get all relevant info
        debug_info = {
            "training_status": training_status,
            "active_tasks": len(asyncio.all_tasks()),
            "llm_initialized": llm is not None,
            "collector_initialized": collector is not None,
            "current_time": datetime.now().isoformat(),
            "server_uptime": "running",  # You can add actual uptime if needed
        }
        print("\n=== Training Debug Info ===")
        print(json.dumps(debug_info, indent=2))
        return debug_info
    except Exception as e:
        return {"error": str(e)}