from llama_index.core.memory import ChatMemoryBuffer
# --- THIS IS THE CRITICAL IMPORT FOR DATA CONSISTENCY ---
from llama_index.core.llms import ChatMessage, MessageRole
from core_components import retriever, qa_template, llm

# --- Initialize components specific to the chat application ---
# The memory buffer is created here because it's tied to a user session.
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

def answer_with_rag(question: str) -> str:
    """
    Answers a question using RAG components with consistent data types.
    """
    if retriever is None or llm is None:
        return "❌ Core RAG components are not initialized. Please check core_components.py."

    try:
        # Step 1: Retrieve the top 3 most relevant nodes directly.
        final_nodes = retriever.retrieve(question)

        # Step 2: Build the prompt using the retrieved context.
        context_str = "\n\n".join([node.get_content() for node in final_nodes])
        final_prompt = qa_template.format(context_str=context_str, question=question)

        # --- Step 3: Add history (CORRECTED) ---
        # 1. Get the history, which is already a list of ChatMessage objects.
        messages = memory.get()
        
        # 2. Append the new user prompt as a ChatMessage object, not a dict.
        messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        # --- END OF FIX ---

        # Step 4: Get the response from the LLM.
        # The 'llm.chat' method now receives a consistent list of ChatMessage objects.
        response = llm.chat(messages)
        response_content = response.message.content

        # Step 5: Update the memory with the new exchange.
        # We continue to use ChatMessage objects for consistency.
        memory.put(ChatMessage(role=MessageRole.USER, content=question))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_content))

        return response_content

    except Exception as e:
        print(f"❌ Chat processing error: {e}")
        return "متاسفانه در پردازش سوال شما خطایی رخ داد."
