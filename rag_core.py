import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.cohere import CohereEmbedding

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
        Settings.llm = Groq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.4
        )
        
        Settings.embed_model = CohereEmbedding(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-multilingual-v3.0",
            input_type="search_query",
        )
        print(f"Models configured successfully for querying.")

    except Exception as e:
        print(f"❌ Error configuring models: {e}")
        return

    qa_prompt_template_str = (
        "شما یک دستیار هوش مصنوعی هستید که به عنوان کارشناس متخصص در 'آیین‌نامه آموزشی دانشگاه سجاد' فعالیت می‌کنید.\n"
        "وظیفه شما پاسخ دقیق و واضح به سوالات دانشجویان بر اساس اطلاعاتی است که در ادامه در بخش 'اطلاعات مرتبط از آیین‌نامه' ارائه شده است.\n\n"
        "**قوانین رفتاری شما (بسیار مهم):**\n"
        "1.  **پایبندی کامل به منبع:** پاسخ‌های شما باید همیشه و فقط بر اساس 'اطلاعات مرتبط از آیین‌نامه' باشد. هرگز، تحت هیچ شرایطی، قانونی را از خود ابداع نکنید.\n"
        "2.  **حفظ حوزه تخصصی:** شما فقط به سوالات مربوط به قوانین آموزشی پاسخ می‌دهید. در صورت پرسیدن سوالات نامرتبط، با احترام بگویید که در این زمینه تخصص ندارید.\n"
        "3.  **لحن حرفه‌ای و رسمی:** با کاربران به صورت رسمی اما مفید و راهگشا صحبت کنید.\n"
        "4. **استناد به منبع:** در صورت امکان، در پاسخ خود به منبع اطلاعات (مثلا شماره صفحه) اشاره کنید تا پاسخ شما معتبرتر باشد.\n\n"
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
    """
    Answers a question using the RAG pipeline.
    """
    if retriever is None:
        return "RAG system not initialized."

    try:
        # Step 1: Retrieve relevant context
        final_nodes = retriever.retrieve(question)

        # This section is updated to include metadata
        # Step 2: Build prompt with context and source information
        context_chunks = []
        for node in final_nodes:
            # Safely get the page number from metadata
            page_number = node.metadata.get('page_label', 'N/A')
            source_info = f"[منبع: صفحه {page_number}]"
            context_chunks.append(f"{source_info}\n{node.get_content()}")
        
        # Join the chunks with a clear separator
        context_str = "\n\n---\n\n".join(context_chunks)
        # --- End of update ---

        final_prompt = qa_template.format(context_str=context_str, question=question)

        # ... The rest of the function remains the same ...
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