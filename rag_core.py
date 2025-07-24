# File: rag_core.py
# This version is adapted for an English-language knowledge base.
# The core logic remains the same, but the prompt is now in English.

import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer

# --- Load environment variables ---
load_dotenv()

# --- Global variables ---
chat_engine = None

def initialize_rag_system():
    """
    Initializes the RAG system with an English prompt template.
    """
    global chat_engine
    
    print("Initializing RAG system for English regulations...")
    
    # --- Configure the models ---
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    try:
        Settings.llm = Groq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )
        print(f"Groq LLM ({Settings.llm.model}) configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Groq LLM: {e}")
        return

    # --- CRITICAL CHANGE: The prompt is now in English ---
    # This prompt defines the persona and strict rules for the regulations assistant.
    qa_prompt_template_str = (
        "You are an AI assistant acting as an expert on the 'Sadjad University Educational Regulations'.\n"
        "Your task is to provide clear and accurate answers to student questions based on the information provided in the 'Relevant Context from the Regulations' section below.\n\n"
        "**Your Behavioral Rules (Very Important):**\n"
        "1.  **Stay on Topic:** You only answer questions related to the educational regulations. If asked about other topics (like history, literature, etc.), politely state that it is outside your area of expertise.\n"
        "2.  **Strictly Adhere to the Source:** Your answers must ALWAYS and ONLY be based on the provided 'Relevant Context'. Never, under any circumstances, invent a rule or provide information not present in the source. If the context does not contain the answer, state that you do not have that information.\n"
        "3.  **Professional Tone:** Address users in a professional, formal, and helpful manner. Formulate your answers as clear and readable paragraphs.\n"
        "4.  **Information Security:** Never mention your data source, file paths, or any internal system metadata.\n\n"
        "---------------------\n"
        "[Relevant Context from the Regulations]\n"
        "{context_str}\n"
        "---------------------\n"
        "[User's Question]\n"
        "{question}\n"
        "---------------------\n"
        "[Your Answer]\n"
    )
    qa_prompt_template = PromptTemplate(qa_prompt_template_str)

    try:
        # Load the pre-built index from the 'storage' directory.
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        
        # Use a stable chat memory buffer.
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        # Create the chat engine with our custom English prompt and memory.
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            text_qa_template=qa_prompt_template,
            verbose=True
        )
        print("✅ RAG system initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing RAG system. Did you run 'ingest.py' successfully? Error: {e}")

def answer_with_rag(question):
    """
    Answers a user's question using the LlamaIndex chat engine.
    """
    if chat_engine is None:
        return "RAG system is not initialized. Please check the server logs for errors."

    try:
        response = chat_engine.chat(question)
        return str(response)
    except Exception as e:
        print(f"❌ Error during chat processing with Groq API: {e}")
        return "An error occurred while processing your question with the online service."

# --- Initialize the system when this module is loaded ---
initialize_rag_system()
