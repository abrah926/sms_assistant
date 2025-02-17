import chromadb
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Load DeepSeek-R1 7B
MODEL_PATH = "/home/abraham/models/deepseek-7b"

def log(message):
    print(f"✅ {message}")

log("🔄 Loading DeepSeek-R1 7B on CPU for responses...")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ Load model with CPU-Only Optimization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

model.eval()
log("✅ DeepSeek-R1 7B is ready for responses!")

# ✅ Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="deepseek_custom")

# ✅ Function to Embed Query
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

# ✅ Function to Retrieve & Generate Response
def retrieve_and_generate(query):
    """Retrieve relevant messages & generate a response."""
    log(f"🔍 Query: {query}")

    # ✅ Generate query embedding
    query_embedding = embed_text(query)

    # ✅ Retrieve top 3 similar messages from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    if not results["metadatas"][0]:  # If no similar examples found
        log("⚠️ No relevant past conversations found. Generating response without context.")
        retrieved_texts = []
    else:
        retrieved_texts = [item["text"] for item in results["metadatas"][0]]

    # ✅ Construct prompt with Bernardo’s persona
    persona = (
        "You are Bernardo Guerra, Co-Founder of Native Materials in San Juan, PR. "
        "You are a smooth but cunning salesman and entrepreneur. Your company imports and sells "
        "construction materials in large quantities, so sales involve large orders. No small order "
        "is taken unless multiple orders combined make a large order. You handle lead acquisition, "
        "customer retention, order processing and fulfillment, sales, and CRM-related tasks. "
        "The customer should never be able to tell the difference between you and Bernardo. "
        "You are charismatic but to the point.\n\n"
    )

    # ✅ Combine retrieved messages as context
    context = " ".join(retrieved_texts[:3])
    prompt = f"{persona}Context: {context}\n\nCustomer: {query}\nBernardo:"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to("cpu")

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=256
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ✅ Interactive Query Test
if __name__ == "__main__":
    while True:
        user_query = input("\n📝 Enter customer query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        response = retrieve_and_generate(user_query)
        print("\n🧠 **DeepSeek-R1 7B Response:**", response)
