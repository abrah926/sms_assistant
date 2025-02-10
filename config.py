import os
from dotenv import load_dotenv
import torch

load_dotenv()

# Matrix Configuration
MATRIX_HOMESERVER_URL = os.getenv("MATRIX_HOMESERVER_URL")
MATRIX_USER_ID = os.getenv("MATRIX_USER_ID")
MATRIX_ACCESS_TOKEN = os.getenv("MATRIX_ACCESS_TOKEN")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://abrah926:Casa1758@localhost/smsdb")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Rate Limiting
RATE_LIMIT_MINUTE = int(os.getenv("RATE_LIMIT_MINUTE", "60"))
RATE_LIMIT_HOUR = int(os.getenv("RATE_LIMIT_HOUR", "1000"))

# Monitoring
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# Add these settings
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "mistralai/Mistral-7B-v0.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add this to your imports and settings
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") 