#!/bin/bash

# 1. Check if already in venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

# 2. Install/update dependencies
pip install -r requirements.txt

# 3. Start monitoring in background
python monitor_training.py &
MONITOR_PID=$!

# 4. Start TensorBoard
tensorboard --logdir ./mistral-finetuned-lora/runs --port 6006 &
TENSOR_PID=$!

# 5. Check system resources
echo "System Check:"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk Space: $(df -h / | awk 'NR==2 {print $4}') available"

# 6. Start training
echo -e "\nStarting LoRA training..."
python prepare_lora.py

# 7. Cleanup on exit
trap "kill $MONITOR_PID $TENSOR_PID" EXIT 