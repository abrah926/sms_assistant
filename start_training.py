import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from training import SalesModelTrainer
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from config import LLM_MODEL_PATH, DATABASE_URL

# Create async engine and session
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session

async def main():
    print("\n=== Starting model training ===")
    
    try:
        print("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            "models/mistral",
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("models/mistral")
        
        trainer = SalesModelTrainer(model, tokenizer)
        
        async with AsyncSessionLocal() as session:
            await trainer.train_iteratively(
                session=session,
                iterations=50
            )
            
    except Exception as e:
        print(f"Error in training: {e}")

if __name__ == "__main__":
    asyncio.run(main())