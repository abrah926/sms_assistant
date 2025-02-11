import os
from dotenv import load_dotenv
import torch

load_dotenv()

# Matrix Configuration
MATRIX_HOMESERVER_URL = os.getenv("MATRIX_HOMESERVER_URL")
MATRIX_USER_ID = os.getenv("MATRIX_USER_ID")
MATRIX_ACCESS_TOKEN = os.getenv("MATRIX_ACCESS_TOKEN")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost/sms_agent")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Rate Limiting
RATE_LIMIT_MINUTE = int(os.getenv("RATE_LIMIT_MINUTE", "60"))
RATE_LIMIT_HOUR = int(os.getenv("RATE_LIMIT_HOUR", "1000"))

# Monitoring
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# Add these settings
def get_device():
    if os.getenv("USE_GPU", "true").lower() == "true":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("Warning: GPU requested but PyTorch CUDA support not available. Using CPU instead.")
    return "cpu"

DEVICE = get_device()
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")  # Use env value first

# Add this to your imports and settings
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN not found in environment variables") 