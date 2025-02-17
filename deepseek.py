from datasets import load_dataset
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor

# ✅ Use DeepSeek-R1 7B instead of 67B
MODEL_PATH = "/home/abraham/models/deepseek-7b"

print("Loading DeepSeek-R1 7B on CPU with optimized RAM usage...")

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
print("✅ DeepSeek-R1 7B is loaded on CPU with OPTIMIZED RAM USAGE!")

# ✅ List of datasets (Sales Conversations FIRST)
datasets_list = [
    "goendalf666/sales-conversations",  # PRIORITY
    "ag_news",
    "AlekseyKorshuk/persona-chat",
    "blended_skill_talk",
    "daily_dialog",
    "multi_woz_v22"
]

# ✅ Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Function to Embed Text (Fixed Output Processing)
def embed_text(text):
    """Generate embeddings using DeepSeek-R1 7B."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=128)
    return outputs[0].cpu().numpy().tolist()

# ✅ Process and Store Dataset in ChromaDB (Parallelized)
def process_dataset(dataset_name):
    """Loads a dataset, embeds it, and stores in ChromaDB."""
    print(f"Processing dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name, cache_dir="~/.cache/huggingface/datasets")
        collection = chroma_client.get_or_create_collection(name=dataset_name.replace("/", "_"))

        batch_size = 10  # Reduce batch size for lower RAM usage
        texts = dataset["train"]["text"]

        def process_batch(start_idx):
            batch = texts[start_idx : start_idx + batch_size]
            embeddings = [embed_text(text) for text in batch]
            collection.add(
                ids=[f"{dataset_name}_{i}" for i in range(start_idx, start_idx + len(batch))],
                embeddings=embeddings,
                metadatas=[{"source": dataset_name, "text": text} for text in batch]
            )
            print(f"✅ Stored batch {start_idx} - {start_idx + len(batch)} for {dataset_name}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            for i in range(0, len(texts), batch_size):
                executor.submit(process_batch, i)

    except Exception as e:
        print(f"⚠️ Failed to process {dataset_name}: {e}")

# ✅ Run Processing for All Datasets in Parallel
for dataset in datasets_list:
    process_dataset(dataset)

# ==============================================
# 🔎 **Step 2: Multi-Dataset RAG Query Retrieval**
# ==============================================
def retrieve_and_generate(query):
    """Retrieve from ALL datasets, prioritizing 'sales-conversations', then generate a response."""
    query_embedding = embed_text(query)
    retrieved_texts = []

    for dataset_name in datasets_list:
        collection_name = dataset_name.replace("/", "_")
        try:
            collection = chroma_client.get_collection(name=collection_name)
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            
            if dataset_name == "goendalf666/sales-conversations":
                retrieved_texts.extend([item["text"] for item in results["metadatas"][0]] * 2)
            else:
                retrieved_texts.extend([item["text"] for item in results["metadatas"][0]])
        except Exception as e:
            print(f"⚠️ No data found for {dataset_name}: {e}")

    context = " ".join(retrieved_texts[:5])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], max_length=500)
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ✅ Example Query
query_text = "How should I handle a customer objection in sales?"
response = retrieve_and_generate(query_text)
print("\n🧠 **DeepSeek-R1 7B Response:**", response)
