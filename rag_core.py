import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

# Load environment variables
load_dotenv()

# Global components
'''
This is the "search engine" or retriever. After being initialized, 
it holds the LlamaIndex object that knows how to search the dataset to find the most relevant parts of the regulations
'''
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
            temperature=0.2
        )
        # You need to be online for the first time to download this model from HuggingFace
        Settings.embed_model = HuggingFaceEmbedding(model_name="heydariai/persian-embeddings") 
        print(f"Models configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring models: {e}")
        return

    qa_prompt_template_str = (
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
    qa_template = PromptTemplate(qa_prompt_template_str)

    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")  # Saving management for llamaindex
        index = load_index_from_storage(storage_context)  # Loading the database 
        retriever = index.as_retriever(similarity_top_k=3)  # Search engine (retiever): It looks for the most similar 3 chunks to the question
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)  # Memory of chatbot: robot can remember last 3000 tokens
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
        # Step 1: Retrieve: check the similarity of question and the 3 chuncks
        final_nodes = retriever.retrieve(question)

        # Step 2: Build prompt
        context_str = "\n\n".join([node.get_content() for node in final_nodes])
        final_prompt = qa_template.format(context_str=context_str, question=question)

        # Step 3: Add history
        # 1. Get the history, which is already a list of ChatMessage objects.
        messages = memory.get()
        
        # 2. Append the new user prompt as a ChatMessage object
        messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))

        # Step 4: Get response from LLM
        response = Settings.llm.chat(messages)
        response_content = response.message.content

        # Step 5: Update memory
        # We must wrap the new messages in ChatMessage objects before putting them into memory.
        memory.put(ChatMessage(role=MessageRole.USER, content=question))  # Memorizing the user's question
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_content))  # Memorizing the chatbot's response

        return response_content

    except Exception as e:
        print(f"❌ Chat processing error: {e}")
        return "متاسفانه در پردازش سوال شما خطایی رخ داد."

# Auto-initialize when imported
initialize_rag_system()
