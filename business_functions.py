from typing import Dict, Optional, List
from decimal import Decimal
from models import Product, Customer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def get_product_price(
    product_name: str,
    quantity_kg: float,
    db: AsyncSession
) -> Dict:
    """Get product price with bulk discounts"""
    query = select(Product).where(Product.name.ilike(f"%{product_name}%"))
    result = await db.execute(query)
    product = result.scalar_one_or_none()
    
    if not product:
        return {
            "error": f"Product {product_name} not found",
            "available_products": ["Steel", "Copper", "Aluminium"]
        }
    
    # Convert to Decimal for precise calculations
    unit_price = Decimal(str(product.price_per_kg))
    quantity = Decimal(str(quantity_kg))
    
    # Apply bulk discounts
    if quantity_kg >= 1000:
        unit_price *= Decimal('0.85')  # 15% discount
    elif quantity_kg >= 500:
        unit_price *= Decimal('0.90')  # 10% discount
    elif quantity_kg >= 200:
        unit_price *= Decimal('0.95')  # 5% discount
    
    total_price = unit_price * quantity
    
    return {
        "product": product.name,
        "quantity_kg": float(quantity),
        "unit_price": float(unit_price),
        "total_price": float(total_price),
        "available_kg": product.available_kg,
        "min_order_kg": product.min_order_kg
    }

async def check_inventory(
    product_name: str,
    db: AsyncSession
) -> Dict:
    """Check product inventory"""
    query = select(Product).where(Product.name.ilike(f"%{product_name}%"))
    result = await db.execute(query)
    product = result.scalar_one_or_none()
    
    if not product:
        return {"error": f"Product {product_name} not found"}
    
    return {
        "product": product.name,
        "available_kg": product.available_kg,
        "min_order_kg": product.min_order_kg
    }

async def process_order(
    product_name: str,
    quantity_kg: float,
    customer_phone: str,
    db: AsyncSession
) -> Dict:
    """Process a new order"""
    # Check product
    query = select(Product).where(Product.name.ilike(f"%{product_name}%"))
    result = await db.execute(query)
    product = result.scalar_one_or_none()
    
    if not product:
        return {"error": f"Product {product_name} not found"}
    
    if quantity_kg < product.min_order_kg:
        return {"error": f"Minimum order is {product.min_order_kg}kg"}
    
    if quantity_kg > product.available_kg:
        return {"error": f"Only {product.available_kg}kg available"}
    
    # Check customer
    customer_query = select(Customer).where(Customer.phone == customer_phone)
    result = await db.execute(customer_query)
    customer = result.scalar_one_or_none()
    
    if not customer:
        return {"error": "Customer not found. Please register first"}
    
    if not customer.payment_info:
        return {"error": "No payment information on file"}
    
    # Calculate price with discounts
    price_info = await get_product_price(product_name, quantity_kg, db)
    
    # TODO: Create Order model and save order
    
    return {
        "status": "success",
        "order_details": {
            "customer": customer.name,
            "product": product.name,
            "quantity_kg": quantity_kg,
            "total_price": price_info["total_price"],
            "payment_method": f"{customer.payment_info['type']} ending in {customer.payment_info['card_ending']}"
        }
    } 