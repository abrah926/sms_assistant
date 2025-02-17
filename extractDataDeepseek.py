from datasets import load_dataset
import json
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor

# ✅ Use DeepSeek-R1 7B instead of 67B
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

# ✅ List of datasets to process
datasets_list = [
    "goendalf666/sales-conversations",
    "AlekseyKorshuk/persona-chat",
    "blended_skill_talk",
    "daily_dialog",
    "multi_woz_v22",
    "ag_news"
]

# ✅ Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def extract_relevant_text(dataset_name, dataset):
    """Extracts sales-related conversations and selects top 100 examples."""
    log(f"Processing dataset: {dataset_name}")
    extracted_texts = []

    try:
        if dataset_name == "goendalf666/sales-conversations":
            for row in dataset["train"]:
                extracted_texts.append(" ".join([v for v in row.values() if v]))
        elif dataset_name == "AlekseyKorshuk/persona-chat":
            for row in dataset["train"]:
                for utterance in row["utterances"]:
                    extracted_texts.append(" ".join(utterance["candidates"]))
        elif dataset_name == "blended_skill_talk":
            for row in dataset["train"]:
                extracted_texts.append(" ".join(row["free_messages"] + row["guided_messages"]))
        elif dataset_name == "daily_dialog":
            for row in dataset["train"]:
                extracted_texts.append(" ".join(row["dialog"]))
        elif dataset_name == "multi_woz_v22":
            for row in dataset["train"]:
                extracted_texts.append(" ".join(row["turns"]["utterance"]))
        elif dataset_name == "ag_news":
            for row in dataset["train"]:
                extracted_texts.append(row["text"])
    except Exception as e:
        log(f"⚠️ Error processing {dataset_name}: {e}")

    # Select top 100 relevant examples
    selected_texts = extracted_texts[:100]  # Simple selection for now
    log(f"Extracted {len(selected_texts)} relevant examples from {dataset_name}.")
    return selected_texts

# ✅ Extract relevant texts from all datasets
final_dataset = []
for dataset_name in datasets_list:
    dataset = load_dataset(dataset_name, cache_dir="~/.cache/huggingface/datasets")
    final_dataset.extend(extract_relevant_text(dataset_name, dataset))

# ✅ Save final dataset
output_path = "DeepSeek_dataset.json"
with open(output_path, "w") as f:
    json.dump(final_dataset, f, indent=4)

log(f"Finished processing all datasets! Saved as {output_path}.")
