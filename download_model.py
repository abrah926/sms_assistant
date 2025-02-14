import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LLM_MODEL_PATH, HUGGINGFACE_TOKEN
import torch

def download_model():
    print(f"\n=== Downloading Model: {LLM_MODEL_PATH} ===")
    
    # Create models directory
    os.makedirs("models/mistral", exist_ok=True)
    
    try:
        print("1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",  # Updated to v0.2
            token=HUGGINGFACE_TOKEN  # Use token from config
        )
        print("✓ Tokenizer downloaded")
        
        print("\n2. Downloading model (this will take a while)...")
        print("Note: Model is about 4.5GB")
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",  # Updated to v0.2
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded")
        
        print("\n3. Saving locally...")
        model.save_pretrained("models/mistral")
        tokenizer.save_pretrained("models/mistral")
        print("✓ Saved to models/mistral/")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure token is loaded:", bool(HUGGINGFACE_TOKEN))
        print("3. Make sure you have enough disk space (~5GB)")
        return None, None

if __name__ == "__main__":
    model, tokenizer = download_model()
    if model and tokenizer:
        print("\n✅ Model ready to use!") 