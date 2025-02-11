from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, Float, JSON, JSONB
from sqlalchemy.ext.declarative import declarative_base
import enum
from datetime import datetime, timezone
import json
from typing import List, Dict
from sqlalchemy.sql import select, func
from sqlalchemy import Index

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
    meta_data = Column(JSON)
    message_type = Column(Enum(MessageType), nullable=False, default=MessageType.GENERAL)

    def to_dict(self):
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "phone": self.phone,
            "content": self.content,
            "direction": self.direction,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_type": self.message_type.value if self.message_type else None,
            "meta_data": self.meta_data
        }

    @classmethod
    async def get_message_history(cls, from_number: str, session) -> List[Dict]:
        """Get message history as list of dicts"""
        query = select(cls).where(cls.phone == from_number)
        result = await session.execute(query)
        messages = result.scalars().all()
        return [message.to_dict() for message in messages]

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

class TrainingExample(Base):
    __tablename__ = "training_examples"
    
    id = Column(Integer, primary_key=True)
    customer_message = Column(String, nullable=False)
    agent_response = Column(String, nullable=False)
    context = Column(String)  # Previous conversation context
    metadata = Column(JSONB)  # Store things like product, intent, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_training_created', created_at),
    ) 