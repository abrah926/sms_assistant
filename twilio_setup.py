from twilio.rest import Client
from fastapi import APIRouter
from pydantic import BaseModel

# Twilio credentials (get from twilio.com)
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
client = Client(account_sid, auth_token)

# Get a Twilio test number
def get_test_number():
    numbers = client.incoming_phone_numbers.list(limit=1)
    return numbers[0].phone_number if numbers else None 