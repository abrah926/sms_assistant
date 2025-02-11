from asyncio import Queue
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class MessageTask:
    phone: str
    content: str
    response_queue: Queue

class MessageQueue:
    def __init__(self):
        self.queue = Queue()
        self.response_queues = {}
    
    async def add_message(self, phone: str, content: str) -> str:
        # Create response queue
        response_queue = Queue()
        self.response_queues[phone] = response_queue
        
        # Add to processing queue
        await self.queue.put(MessageTask(phone, content, response_queue))
        
        # Wait for response
        response = await response_queue.get()
        del self.response_queues[phone]
        return response

    async def process_messages(self, llm):
        while True:
            try:
                task = await self.queue.get()
                response = await llm.generate(task.content, [], None)
                await task.response_queue.put(response)
            except Exception as e:
                print(f"Queue processing error: {e}")
                await task.response_queue.put("Error processing message") 