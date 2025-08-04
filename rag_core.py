import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.cohere import Cohere
from llama_index.core.llms import ChatMessage, MessageRole
# This is the corrected import path for the non-deprecated class
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

# Load environment variables
load_dotenv()

# Global components
retriever = None 
memory = None
qa_template = None

def initialize_rag_system():
    global retriever, memory, qa_template
    print("Initializing RAG System...")
    
    # Configure the models
    try:
        Settings.llm = Cohere(
            model="command-r-plus",
            api_key=os.getenv("COHERE_API_KEY"),
            temperature=0.1
        )
        
        # Configure the embedding model to use BAAI/bge-m3 via Hugging Face API
        Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
            token=os.getenv("HUGGINGFACE_API_KEY"),
            model_name="BAAI/bge-m3"
        )
        print(f"Models configured successfully for querying.")

    except Exception as e:
        print(f"❌ Error configuring models: {e}")
        return

    # --- Final and Strengthened Prompt Template ---
    qa_prompt_template_str = (
    "شما یک دستیار هوش مصنوعی هستید که به عنوان کارشناس متخصص در 'آیین‌نامه آموزشی دانشگاه سجاد' فعالیت می‌کنید.\n"
    "وظیفه شما پاسخ دقیق و واضح به سوالات دانشجویان فقط و فقط بر اساس اطلاعاتی است که در بخش 'اطلاعات مرتبط از آیین‌نامه' ارائه شده است.\n\n"
    "**قوانین رفتاری شما (بسیار مهم):**\n"
    "1.  **پایبندی مطلق به منبع:** پاسخ‌های شما باید همیشه و فقط بر اساس 'اطلاعات مرتبط از آیین‌نامه' باشد. هرگز قانونی را از خود ابداع نکنید یا از دانش عمومی خود استفاده ننمایید.\n"
    "2.  **نقل قول و سپس نتیجه‌گیری:** برای پاسخ به هر سوال، ابتدا **جمله دقیق و کامل مربوطه را از منبع نقل قول کن**. سپس، در یک پاراگراف جدید، بر اساس آن نقل قول، نتیجه‌گیری نهایی خود را به صورت واضح و خلاصه بیان کن.\n"
    "3.  **مدیریت اطلاعات ناموجود:** اگر پاسخ سوال در متن ارائه شده وجود ندارد، به وضوح بگو که 'اطلاعات مشخصی در این مورد در آیین‌نامه یافت نشد' و کاربر را به گروه آموزشی ارجاع بده.\n"
    "4.  **لحن حرفه‌ای و رسمی:** با کاربران به صورت رسمی اما مفید و راهگشا صحبت کنید.\n"
    "5. **استناد به منبع:** در پاسخ خود به منبع اطلاعات (مثلا شماره صفحه) اشاره کنید.\n\n"
    "---------------------\n"
    "[اطلاعات مرتبط از آیین‌نامه]\n"
    "{context_str}\n"
    "---------------------\n"
    "[سوال کاربر]\n"
    "{question}\n"
    "---------------------\n"
    "[پاسخ شما]\n"
    )
    qa_template = PromptTemplate(qa_prompt_template_str)

    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=3)
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        print("✅ RAG system components initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")

def answer_with_rag(question: str) -> str:
    if retriever is None:
        return "RAG system not initialized."

    try:
        final_nodes = retriever.retrieve(question)

        context_chunks = []
        for node in final_nodes:
            page_number = node.metadata.get('page_number', 'N/A')
            source_info = f"[منبع: صفحه {page_number}]"
            context_chunks.append(f"{source_info}\n{node.get_content()}")
        
        context_str = "\n\n---\n\n".join(context_chunks)

        final_prompt = qa_template.format(context_str=context_str, question=question)

        messages = memory.get()
        messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))

        response = Settings.llm.chat(messages)
        response_content = response.message.content

        memory.put(ChatMessage(role=MessageRole.USER, content=question)) 
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_content))

        return response_content

    except Exception as e:
        print(f"❌ Chat processing error: {e}")
        return "متاسفانه در پردازش سوال شما خطایی رخ داد."

# Auto-initialize when imported
initialize_rag_system()
