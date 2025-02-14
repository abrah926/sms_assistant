import torch
import os
from datetime import datetime

def show_best_responses():
    checkpoint_dir = "checkpoints"
    print(f"Looking for checkpoints in: {os.path.abspath(checkpoint_dir)}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist!")
        return
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith("checkpoint_") and f.endswith(".pt")]
    
    print(f"Found {len(checkpoints)} checkpoint files: {checkpoints}")
    
    if not checkpoints:
        print("No checkpoints found")
        return
        
    latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    best_responses = checkpoint.get('best_responses', {})
    
    print(f"\n=== Best Responses from Latest Checkpoint ===")
    print(f"Total best responses: {len(best_responses)}")
    
    # Show top responses
    for i, (response, score) in enumerate(sorted(best_responses.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:5], 1):
        print(f"\nTop Response {i} (Score: {score*100:.2f}%):")
        print(f"Response: {response}")
        print("-" * 50)

def show_checkpoint_details():
    checkpoint_dir = "checkpoints"
    print(f"\n=== Checkpoint Details ===")
    print(f"Looking in: {os.path.abspath(checkpoint_dir)}")
    
    if not os.path.exists(checkpoint_dir):
        print("❌ Checkpoint directory does not exist!")
        return
        
    # Check main checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) 
                        if f.startswith("checkpoint_") and f.endswith(".pt")])
    
    print(f"\nMain Checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        path = os.path.join(checkpoint_dir, cp)
        size = os.path.getsize(path) / (1024*1024)  # Size in MB
        timestamp = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"- {cp}")
        print(f"  Size: {size:.1f}MB")
        print(f"  Modified: {timestamp}")
        
        # Load checkpoint data
        try:
            data = torch.load(path)
            print(f"  Iteration: {data.get('iteration', 'unknown')}")
            print(f"  Accuracy: {data.get('running_accuracy', 0)*100:.2f}%")
            print(f"  Best responses: {len(data.get('best_responses', {}))}")
        except Exception as e:
            print(f"  Error loading: {e}")
        print()
    
    # Check backups
    backup_dir = os.path.join(checkpoint_dir, "backups")
    if os.path.exists(backup_dir):
        backups = sorted([f for f in os.listdir(backup_dir) 
                         if f.startswith("backup_") and f.endswith(".pt")])
        print(f"\nBackups ({len(backups)}):")
        for bk in backups:
            path = os.path.join(backup_dir, bk)
            size = os.path.getsize(path) / (1024*1024)
            timestamp = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"- {bk}")
            print(f"  Size: {size:.1f}MB")
            print(f"  Modified: {timestamp}")

if __name__ == "__main__":
    show_checkpoint_details() 