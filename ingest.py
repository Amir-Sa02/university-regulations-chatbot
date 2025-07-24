import os
import re
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    print("Starting data ingestion with specialized Persian embedding model...")

    # Define the directories
    DATA_DIR = "./data"
    STORAGE_DIR = "./storage"

    # Create directories if they don't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}. Please add your document(s) here.")
        return
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    # --- Configure the LlamaIndex Settings ---
    # The sentence-transformers library (used by HuggingFaceEmbedding) will automatically
    # cache this model locally after the first download.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="heydariAI/persian-embeddings"
    )
    print(f"Embedding model configured ({Settings.embed_model.model_name}). This runs locally on your CPU.")

    # --- Load and Process the Persian PDF ---
    print(f"Loading documents from '{DATA_DIR}'...")
    try:
        reader = SimpleDirectoryReader(DATA_DIR)
        raw_documents = reader.load_data()
        if not raw_documents:
            print(f"No documents found in '{DATA_DIR}'.")
            return
        full_text = "\n".join([doc.text for doc in raw_documents])
        print("Successfully loaded document text.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # --- Custom Splitting based on the Persian word "ماده" ---
    pattern = r'(?=ماده\s)'
    text_chunks = re.split(pattern, full_text)
    meaningful_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
    final_documents = [Document(text=chunk) for chunk in meaningful_chunks]
    
    print(f"Successfully split the document into {len(final_documents)} meaningful chunks based on 'ماده'.")

    # --- Create and Persist the Index ---
    print("Creating vector index... This may take a moment.")
    index = VectorStoreIndex.from_documents(final_documents, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    print(f"Data ingestion complete. The index has been created and saved in the '{STORAGE_DIR}' folder.")

if __name__ == "__main__":
    main()
