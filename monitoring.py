from prometheus_client import Counter, Histogram
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Metrics
message_counter = Counter(
    'sms_messages_total',
    'Number of SMS messages processed',
    ['direction', 'type']
)

error_counter = Counter(
    'sms_errors_total',
    'Number of errors encountered',
    ['type']
)

response_time = Histogram(
    'sms_response_time_seconds',
    'Response time in seconds'
)

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response_time.observe(process_time)
        return response 