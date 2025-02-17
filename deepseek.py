import json
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
    torch_dtype=torch.float32,  # Prevents float16 CPU instability
    device_map="cpu",  # Force CPU mode
    low_cpu_mem_usage=True,  # Load layer-by-layer instead of full model in RAM
    offload_folder="offload_dir"  # Moves unused layers to disk to free RAM
)

model.eval()  # Set to evaluation mode
log("DeepSeek-R1 7B is loaded on CPU with OPTIMIZED RAM USAGE!")

# ✅ Load Custom Dataset
DATASET_PATH = "DeepSeek_dataset.json"
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

log(f"Loaded custom dataset from {DATASET_PATH}, containing {len(dataset)} examples.")

# ✅ Initialize ChromaDB (Persistent Storage)
# ✅ Initialize ChromaDB (Ensure Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Delete old collection to avoid mismatched dimensions
try:
    chroma_client.delete_collection(name="deepseek_custom")
    print("✅ Old ChromaDB collection deleted.")
except Exception as e:
    print(f"⚠️ Warning: {e}")

# ✅ Recreate collection with correct dimension (465)
collection = chroma_client.get_or_create_collection(name="deepseek_custom", metadata={"dim": 465})
print("✅ ChromaDB collection reset with dimension 465")

print(f"✅ ChromaDB collection created with dimension: {collection.metadata}")

# ✅ Function to Embed Text
def embed_text(text):
    """Generate embeddings using DeepSeek-R1 7B."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to("cpu", non_blocking=True)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=128
        )

    embedding = outputs[0].cpu().numpy().tolist()
    
    # ✅ Print embedding size to confirm dimension
    print(f"✅ Debug: Generated embedding dimension: {len(embedding)}")
    
    return embedding



# ✅ Process Dataset
log("Embedding custom dataset into ChromaDB...")
for i, text in enumerate(dataset):
    embedding = embed_text(text)
    collection.add(
        ids=[f"example_{i}"],
        embeddings=[embedding],
        metadatas=[{"source": "custom", "text": text}]
    )
    if i % 10 == 0:
        log(f"Processed {i}/{len(dataset)} examples...")

log("Finished embedding dataset into ChromaDB!")

# ✅ Retrieval & Response Generation
def retrieve_and_generate(query):
    """Retrieve relevant texts and generate a response."""
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    retrieved_texts = [item["text"] for item in results["metadatas"][0]]
    
    # ✅ Construct prompt with Bernardo's identity
    persona = (
        "You are Bernardo Guerra, Co-Founder of Native Materials in San Juan, PR. "
        "You are a smooth but cunning salesman and entrepreneur. Your company imports and sells "
        "construction materials in large quantities, so sales incur in large orders. No small order "
        "will be taken unless multiple orders combined make a large order. You handle lead acquisition, "
        "customer retention, order processing and fulfillment, sales, and CRM-related tasks. "
        "The customer should never be able to tell the difference between you and Bernardo. "
        "You are charismatic but to the point.\n\n"
    )
    
    context = " ".join(retrieved_texts[:5])
    prompt = f"{persona}Context: {context}\n\nCustomer: {query}\nBernardo:"
    
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to("cpu")
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_length=500
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ✅ Example Query
test_query = "I need 50 bags of cement, what's the best price you can offer?"
response = retrieve_and_generate(test_query)
log("Generated response:")
print("\n🧠 **DeepSeek-R1 7B Response:**", response)
