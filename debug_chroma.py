from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# ✅ Load DeepSeek Model
MODEL_PATH = "/home/abraham/models/deepseek-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()

# ✅ Short test sentence
test_text = "This is a test sentence to check the embedding size."

def embed_text_debug(text):
    """Generate embeddings using DeepSeek-R1 7B and print their shape."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to("cpu")

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )

    # ✅ Extract last hidden state (4096 dimensions)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

    # ✅ Truncate to first 465 values (DeepSeek uses 4096, ChromaDB needs 465)
    reduced_embedding = embedding[:465]

    print(f"✅ Debug: Truncated embedding dimension: {len(reduced_embedding)}")  # Should be 465

    return reduced_embedding

# ✅ Run Debug Test
embed_text_debug(test_text)
