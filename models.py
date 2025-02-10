from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
import enum
from datetime import datetime, timezone

Base = declarative_base()

class MessageType(enum.Enum):
    QUESTION = "question"
    ORDER = "order"
    SUPPORT = "support"
    GENERAL = "general"

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    phone = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    direction = Column(String, nullable=False)  # incoming/outgoing
    timestamp = Column(DateTime(timezone=True), nullable=False)
    meta_data = Column(Text, nullable=True)
    message_type = Column(Enum(MessageType), nullable=False, default=MessageType.GENERAL)

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    price_per_kg = Column(Float, nullable=False)
    min_order_kg = Column(Float, nullable=False)
    available_kg = Column(Float, nullable=False)

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True, nullable=False)
    name = Column(String)
    payment_info = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc)) 