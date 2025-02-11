from datasets import load_dataset

# Try loading one of our datasets
try:
    print("Loading daily_dialog dataset...")
    ds = load_dataset("daily_dialog", split="train")
    print(f"Successfully loaded {len(ds)} examples")
except Exception as e:
    print(f"Error loading dataset: {e}") 