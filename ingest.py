import os
import json
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
)
# This is the corrected import path for the non-deprecated class
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

load_dotenv()

def main():
    print("Starting data ingestion from JSONL file...")

    JSONL_FILE_PATH = "./data/regulations.jsonl"
    STORAGE_DIR = "./storage"

    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    # 1. Configure Settings to use BAAI/bge-m3 via Hugging Face API
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        token=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="BAAI/bge-m3"
    )
    print(f"Embedding model configured: {Settings.embed_model.model_name}")

    # 2. Load Documents directly from JSONL file
    documents = []
    with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                text=data["text"],
                metadata=data["metadata"]
            )
            documents.append(doc)
    
    print(f"Successfully loaded and created {len(documents)} documents from JSONL.")
    
    if not documents:
        print(f"No documents found in '{JSONL_FILE_PATH}'.")
        return

    # 3. Create and Persist the Index
    print("Creating vector index...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    print(f"✅ Data ingestion complete and saved in '{STORAGE_DIR}'.")

if __name__ == "__main__":
    main()
