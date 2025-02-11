from fastapi import HTTPException
import re
from typing import Optional
import json
from models import MessageType
from datetime import datetime, timezone

def sanitize_phone_number(phone: str) -> str:
    """Remove everything except digits and '+' from phone numbers"""
    return re.sub(r'[^\d+]', '', phone)

def sanitize_message(message: str) -> str:
    """Basic message sanitization"""
    # Remove any control characters
    message = ''.join(char for char in message if ord(char) >= 32)
    return message.strip()

def classify_message_type(message: str) -> MessageType:
    """Classify message type based on content"""
    message = message.lower()
    
    if any(word in message for word in ['buy', 'order', 'purchase', 'price']):
        return MessageType.ORDER
    elif any(word in message for word in ['help', 'support', 'issue', 'problem']):
        return MessageType.SUPPORT
    elif '?' in message or any(word in message for word in ['what', 'how', 'when', 'where', 'why']):
        return MessageType.QUESTION
    return MessageType.GENERAL

class MessageMetadata:
    def __init__(self, message: str):
        self.message = message
        self.timestamp = datetime.now(timezone.utc)
        self.language = self._detect_language(message)
        self.urgency = self._assess_urgency(message)
        self.sentiment = self._analyze_sentiment(message)
    
    def _detect_language(self, message: str) -> str:
        """Simple language detection"""
        # For now, default to English
        return "en"
    
    def _assess_urgency(self, message: str) -> str:
        """Simple urgency assessment"""
        urgent_words = ['urgent', 'asap', 'emergency', 'immediately']
        message = message.lower()
        if any(word in message for word in urgent_words):
            return "high"
        return "normal"
    
    def _analyze_sentiment(self, message: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'thanks', 'please']
        negative_words = ['bad', 'poor', 'issue', 'problem', 'wrong']
        
        message = message.lower()
        pos_count = sum(1 for word in positive_words if word in message)
        neg_count = sum(1 for word in negative_words if word in message)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"
    
    def to_json(self) -> dict:
        return {
            'raw_message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'length': len(self.message),
            'language': self.language,
            'urgency': self.urgency,
            'sentiment': self.sentiment
        }

def get_fallback_response(prompt: str) -> str:
    if "price" in prompt.lower() and "copper" in prompt.lower():
        return "Copper is $8,500/ton with a 1 ton minimum. Would you like to place an order?"
    elif "steel" in prompt.lower():
        return "Steel is $800/ton with 5 ton minimum. How many tons do you need?"
    elif "aluminum" in prompt.lower():
        return "Aluminum is $2,400/ton with 2 ton minimum. Can I help you place an order?"
    else:
        return "How can I help you with your metal order today? We have steel, copper, and aluminum available."