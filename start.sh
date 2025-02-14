#!/bin/bash

# 1. Install/upgrade dependencies
pip install --upgrade pip
pip install transformers accelerate peft bitsandbytes fastapi uvicorn sqlalchemy psycopg2-binary

# 2. Download model if not exists
if [ ! -d "/mnt/storage/mistral" ]; then
    echo "Downloading Mistral-7B model..."
    huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir /mnt/storage/mistral
fi

# 3. Run LoRA training
echo "Starting LoRA training..."
python prepare_lora.py

# 4. Start API server after training
echo "Starting API server..."
uvicorn app:app --host 0.0.0.0 --port 8000 