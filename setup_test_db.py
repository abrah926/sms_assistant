import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from models import Base, Product, Customer
from config import DATABASE_URL
from datetime import datetime, timezone

async def setup_test_data():
    engine = create_async_engine(DATABASE_URL)
    
    # Drop and recreate all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        # Add test products
        products = [
            Product(
                name="Steel",
                description="High-quality structural steel",
                price_per_kg=2.50,
                min_order_kg=100,
                available_kg=10000
            ),
            Product(
                name="Copper",
                description="Pure copper, ideal for electrical",
                price_per_kg=8.75,
                min_order_kg=50,
                available_kg=5000
            ),
            Product(
                name="Aluminium",
                description="Lightweight aluminium",
                price_per_kg=4.25,
                min_order_kg=75,
                available_kg=7500
            )
        ]
        
        # Add test customer
        customer = Customer(
            phone="+1234567890",
            name="Abraham",
            payment_info={"type": "visa", "card_ending": "4321"},
            created_at=datetime.now(timezone.utc)
        )
        
        session.add_all(products)
        session.add(customer)
        await session.commit()
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(setup_test_data()) 