import asyncio
from training import SalesModelTrainer
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

async def test_training_pipeline():
    print("\n=== Testing Training Pipeline ===")
    
    # Setup
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        # Load model
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH)
        
        trainer = SalesModelTrainer(model, tokenizer)
        
        async with async_session() as session:
            # Run one training iteration
            await trainer.train_iteratively(session, iterations=1)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_training_pipeline()) 