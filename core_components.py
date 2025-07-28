# This central module initializes and configures all the core components of the RAG

import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import PromptTemplate, load_index_from_storage, StorageContext


# Load environment variables
load_dotenv()

# Global components
retriever = None
qa_template = None
llm = None


def initialize_components():
    """
    Initializes all core RAG components and makes them available globally.
    """

    global retriever, qa_template, llm
    print("Initializing core RAG Components...")

    try:
        # 1. Configure the LLM (Generator)
        llm = Groq(
            model = "llama-3.1-8b-instant",
            api_key= os.getenv("GROQ_API_KEY"),
            temperature = 0.2
        )

        # 2. Configure the Embedding model (For retriever)
        Settings.embed_model = HuggingFaceEmbedding(model_name="heydariai/persian-embeddings")
        
        print(f"✅ Models configured successfully.")

    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        return
    
    # 3. Define prompt template
    qa_prompt_template = (
        "شما یک دستیار هوش مصنوعی هستید که به عنوان کارشناس متخصص در 'آیین‌نامه آموزشی دانشگاه سجاد' فعالیت می‌کنید.\n"
        "وظیفه شما پاسخ دقیق و واضح به سوالات دانشجویان بر اساس اطلاعاتی است که در ادامه در بخش 'اطلاعات مرتبط از آیین‌نامه' ارائه شده است.\n\n"
        "**قوانین رفتاری شما (بسیار مهم):**\n"
        "1.  **پایبندی کامل به منبع:** پاسخ‌های شما باید همیشه و فقط بر اساس 'اطلاعات مرتبط از آیین‌نامه' باشد. هرگز، تحت هیچ شرایطی، قانونی را از خود ابداع نکنید.\n"
        "2.  **حفظ حوزه تخصصی:** شما فقط به سوالات مربوط به قوانین آموزشی پاسخ می‌دهید.\n"
        "3.  **لحن حرفه‌ای و رسمی:** با کاربران به صورت رسمی اما مفید و راهگشا صحبت کنید.\n\n"
        "---------------------\n"
        "[اطلاعات مرتبط از آیین‌نامه]\n"
        "{context_str}\n"
        "---------------------\n"
        "[سوال کاربر]\n"
        "{question}\n"
        "---------------------\n"
        "[پاسخ شما]\n"
    )
    qa_template = PromptTemplate(qa_prompt_template)
    
    # 4. Load the index and create the Retriever
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")  # Saving management for llamaindex
        index = load_index_from_storage(storage_context)  # Loading the database 
        retriever = index.as_retriever(similarity_top_k=3)  # Search engine (retiever): It looks for the most similar 3 chunks to the question
        
        print("✅ RAG system components initialized successfully.")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
   
# Auto-initialize when this module is first imported
initialize_components()
