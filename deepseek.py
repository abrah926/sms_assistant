import json
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# ✅ Use DeepSeek-R1 7B
MODEL_PATH = "/home/abraham/models/deepseek-7b"

def log(message):
    print(f"✅ {message}")

log("Loading DeepSeek-R1 7B on CPU with optimized RAM usage...")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ Load model with CPU-Only Optimization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    offload_folder="offload_dir"
)

model.eval()
log("✅ DeepSeek-R1 7B is loaded on CPU with OPTIMIZED RAM USAGE!")

# ✅ Load Custom Dataset
custom_dataset_path = "DeepSeek_dataset.json"

with open(custom_dataset_path, "r") as f:
    final_dataset = json.load(f)

log(f"✅ Loaded custom dataset from {custom_dataset_path}, containing {len(final_dataset)} examples.")

# ✅ Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Reset Collection for Custom Dataset
try:
    chroma_client.delete_collection(name="deepseek_custom")
    log("✅ Old ChromaDB collection deleted.")
except Exception as e:
    log(f"⚠️ Warning: {e}")

collection = chroma_client.get_or_create_collection(name="deepseek_custom", metadata={"dim": 465})
log("✅ ChromaDB collection recreated with dimension 465.")

# ✅ Function to Embed Text
def embed_text(text):
    """Generate embeddings using DeepSeek-R1 7B."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to("cpu", non_blocking=True)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )

    # ✅ Extract last hidden state (4096 dimensions)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

    # ✅ Ensure embeddings are always 465-dimensional
    if len(embedding) > 465:
        embedding = embedding[:465]  # Trim excess dimensions
    elif len(embedding) < 465:
        embedding = list(embedding) + [0] * (465 - len(embedding))  # Pad with zeros

    return embedding

# ✅ Embed & Store in ChromaDB
log("🚀 Starting embedding process for custom dataset...")

start_time = time.time()

for i, text in enumerate(final_dataset):
    embedding = embed_text(text)
    collection.add(
        ids=[f"example_{i}"],
        embeddings=[embedding],
        metadatas=[{"source": "custom_dataset", "text": text}]
    )

    if i % 50 == 0:
        elapsed_time = time.time() - start_time
        log(f"✅ Processed {i}/{len(final_dataset)} examples in {elapsed_time:.2f}s...")

total_elapsed_time = time.time() - start_time
log(f"✅ ✅ Finished embedding custom dataset in {total_elapsed_time:.2f}s! 🎉")
