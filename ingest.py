import os
import re
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
    SimpleDirectoryReader,
)
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

def main():
    print("Starting data ingestion...")

    DATA_DIR = "./data"
    STORAGE_DIR = "./storage"

    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    # 1. Configure Settings
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-multilingual-v3.0",
        input_type="search_document",
    )
    print(f"Embedding model configured: {Settings.embed_model.model_name}")

    # 2. Load Documents with Metadata
    reader = SimpleDirectoryReader(DATA_DIR)
    documents = reader.load_data()
    print(f"Successfully loaded {len(documents)} page(s)(documents).")
    
    if not documents:
        print(f"No documents found in '{DATA_DIR}'.")
        return

    # 3. Custom Splitting while Preserving Metadata
    # We combine your reliable regex splitting with the new metadata-aware loading.
    final_documents = []
    pattern = r'(?=ماده\s\d+)'
    
    # We loop through each page (which is one 'Document' object)
    for doc in documents:
        page_text = doc.text
        
        # Split the page text using the regex
        text_chunks = re.split(pattern, page_text)
        
        # Create a new Document for each chunk and preserve the original metadata
        for chunk in text_chunks:
            if chunk.strip():
                new_doc = Document(text=chunk.strip(), metadata=doc.metadata)
                final_documents.append(new_doc)
    
    print(f"Successfully split the document into {len(final_documents)} chunks with metadata.")

    # 4. Create and Persist the Index from the new documents
    # We now use our manually created 'final_documents' list.
    print("Creating vector index...")
    index = VectorStoreIndex.from_documents(final_documents,show_progress=True)
    
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    print(f"✅ Data ingestion complete and saved in '{STORAGE_DIR}'.")

if __name__ == "__main__":
    main()