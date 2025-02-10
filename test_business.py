import asyncio
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Product, Customer
from business_functions import get_product_price, check_inventory, process_order
from decimal import Decimal
from config import DATABASE_URL

# Test database setup
engine = create_async_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def setup_test_data(db: AsyncSession):
    """Initialize test data"""
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
        payment_info={"type": "visa", "card_ending": "4321"}
    )
    
    db.add_all(products)
    db.add(customer)
    await db.commit()

async def test_get_product_price():
    async with TestingSessionLocal() as db:
        # Test regular price
        result = await get_product_price("Steel", 150, db)
        assert result["product"] == "Steel"
        assert result["unit_price"] == 2.50
        assert result["total_price"] == 375.0  # 150 * 2.50
        
        # Test bulk discount (5%)
        result = await get_product_price("Steel", 200, db)
        assert result["unit_price"] == 2.375  # 2.50 * 0.95
        
        # Test not found
        result = await get_product_price("Invalid", 100, db)
        assert "error" in result

async def test_check_inventory():
    async with TestingSessionLocal() as db:
        result = await check_inventory("Copper", db)
        assert result["product"] == "Copper"
        assert result["available_kg"] == 5000
        assert result["min_order_kg"] == 50

async def test_process_order():
    async with TestingSessionLocal() as db:
        # Test successful order
        result = await process_order("Aluminium", 100, "+1234567890", db)
        assert result["status"] == "success"
        assert result["order_details"]["product"] == "Aluminium"
        
        # Test minimum order
        result = await process_order("Steel", 50, "+1234567890", db)
        assert "error" in result
        assert "Minimum order" in result["error"]
        
        # Test unknown customer
        result = await process_order("Steel", 100, "+9999999999", db)
        assert "error" in result
        assert "Customer not found" in result["error"]

async def run_tests():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as db:
        await setup_test_data(db)
        
        print("\nTesting get_product_price...")
        await test_get_product_price()
        
        print("Testing check_inventory...")
        await test_check_inventory()
        
        print("Testing process_order...")
        await test_process_order()
        
        print("\nAll tests passed! ✅")

if __name__ == "__main__":
    asyncio.run(run_tests()) 