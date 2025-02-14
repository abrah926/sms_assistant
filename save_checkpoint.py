import torch
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_emergency_checkpoint():
    print("\n=== Saving Emergency Checkpoint ===")
    
    try:
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"checkpoints/emergency_checkpoint_{timestamp}"
        os.makedirs(checkpoint_path, exist_ok=True)
        
        print("1. Saving model state...")
        model = AutoModelForCausalLM.from_pretrained("models/mistral")
        tokenizer = AutoTokenizer.from_pretrained("models/mistral")
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata
        checkpoint_meta = {
            'timestamp': timestamp,
            'saved_at': datetime.now().isoformat(),
            'for_lora': True  # Flag for LoRA training
        }
        
        torch.save(checkpoint_meta, f"{checkpoint_path}/meta.pt")
        print(f"\n✓ Checkpoint saved to: {checkpoint_path}")
        print("Ready for LoRA training on cloud!")
        
    except Exception as e:
        print(f"\n❌ Error saving checkpoint: {e}")

if __name__ == "__main__":
    save_emergency_checkpoint() 