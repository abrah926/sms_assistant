import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from config import DATABASE_URL

async def test_connection():
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
            print("Database connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_connection()) 