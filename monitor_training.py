import time
import json
from pathlib import Path
import psutil

def monitor_training():
    print("\n=== Training Monitor Started (CPU Mode) ===")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    while True:
        try:
            # CPU Stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Training Progress
            if Path("logs/best_responses.jsonl").exists():
                with open("logs/best_responses.jsonl") as f:
                    responses = [json.loads(line) for line in f]
                    latest = responses[-1] if responses else None
                    if latest:
                        print(f"\nLatest Checkpoint ({latest['checkpoint_step']:,}):")
                        print(f"Score: {latest['scores']['total']*100:.1f}%")
                        print(f"Loss: {latest['training_loss']:.4f}")
            
            # System Usage
            print("\nSystem Status:")
            print(f"CPU Usage: {cpu_percent}%")
            print(f"Memory: {memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB ({memory.percent}%)")
            
            time.sleep(60)  # Update every minute
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_training() 