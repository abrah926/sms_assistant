from fastapi import HTTPException
import re
from typing import Optional
import json
from models import MessageType

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
        self.language = self.detect_language(message)
        self.urgency = self.assess_urgency(message)
        self.sentiment = self.analyze_sentiment(message)
    
    def to_json(self) -> str:
        return json.dumps({
            'language': self.language,
            'urgency': self.urgency,
            'sentiment': self.sentiment
        }) 