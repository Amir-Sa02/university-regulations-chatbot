import os
from dotenv import load_dotenv
from core_components import retriever, llm
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# --- Load environment variables ---
load_dotenv()

def main():
    """
    Main function to run the comprehensive evaluation process.
    """
    print("🚀 Starting RAG System Evaluation (Official Documentation Method)...")

    if retriever is None or llm is None:
        print("❌ Core RAG components not initialized. Please check core_components.py.")
        return

    # --- 1. Define our Golden Evaluation Dataset ---
    eval_dataset = [
        {
            "query": "حداقل نمره قبولی در هر درس چقدر است؟",
            "expected_context_substring": "حداقل نمره قبولی در هر درس ۱۰ است"
        },
        {
            "query": "مدت مجاز مرخصی زایمان چند نیمسال است؟",
            "expected_context_substring": "مدت مجاز مرخصی زایمان، دو نیمسال تحصیلی"
        },
        {
            "query": "اگر دانشجو مشروط شود، در نیمسال بعد حداکثر چند واحد می‌تواند اخذ کند؟",
            "expected_context_substring": "حداکثر میتواند تا ۱۴ واحد درسی انتخاب کند"
        },
        {
            "query": "آیا تغییر رشته از دوره شبانه به روزانه مجاز است؟",
            "expected_context_substring": "از شبانه به روزانه ، از غیر حضوری به حضوری و نیمه حضوری ممنوع است"
        },
        {
            "query": "پایتخت ایران کجاست؟", # Out-of-domain question
            "expected_context_substring": ""
        }
    ]
    print(f"Loaded {len(eval_dataset)} evaluation cases.")

    # --- 2. Initialize the Evaluators ---
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    print("Evaluators (Faithfulness, Relevancy) initialized.")

    # --- 3. Run the Evaluation Loop ---
    print("\n" + "="*60)
    print("🔬 Running Comprehensive Evaluations...")
    print("="*60 + "\n")
    
    retrieval_results = []
    faithfulness_results = []
    relevancy_results = []

    for i, case in enumerate(eval_dataset, 1):
        question = case["query"]
        expected_context_str = case["expected_context_substring"]
        
        print(f"--- Evaluating Case #{i}: {question} ---")

        # a. Retrieval Step
        retrieved_nodes = retriever.retrieve(question)
        retrieved_context_list = [node.get_content() for node in retrieved_nodes]
        retrieved_context_full_str = "\n\n".join(retrieved_context_list)

        # b. Retrieval Evaluation (Hit Rate)
        is_hit = expected_context_str in retrieved_context_full_str if expected_context_str else not retrieved_context_full_str
        retrieval_results.append(is_hit)
        print(f"🎯 Retrieval Result: {'HIT' if is_hit else 'MISS'}")

        # c. Generation Step
        prompt = f"Context: {retrieved_context_full_str}\n\nQuestion: {question}\n\nAnswer:"
        response_obj = llm.complete(prompt)
        response_text = response_obj.text
        print(f"💬 Generated Response: {response_text}")

        # --- d. Response Evaluation (Using the correct 'evaluate' method) ---
        
        # Faithfulness Evaluation
        faithfulness_result = faithfulness_evaluator.evaluate(
            response=response_text, 
            contexts=retrieved_context_list # Pass the list of context strings
        )
        faithfulness_results.append(faithfulness_result.passing)
        print(f"⚖️ Faithfulness Result: {'PASS' if faithfulness_result.passing else 'FAIL'}")

        # Relevancy Evaluation
        relevancy_result = relevancy_evaluator.evaluate(
            query=question, 
            response=response_text, 
            contexts=retrieved_context_list # Pass the list of context strings
        )
        relevancy_results.append(relevancy_result.passing)
        print(f"📈 Relevancy Result: {'PASS' if relevancy_result.passing else 'FAIL'}")
        
        print("-" * 60 + "\n")

    # --- 4. Print Final Aggregated Report ---
    print("="*60)
    print("📊 Final Evaluation Report")
    print("="*60)
    
    retrieval_pass_rate = (sum(retrieval_results) / len(retrieval_results)) * 100
    print(f"Retrieval Hit Rate: {retrieval_pass_rate:.2f}%")
    
    faithfulness_pass_rate = (sum(faithfulness_results) / len(faithfulness_results)) * 100
    print(f"Faithfulness Pass Rate: {faithfulness_pass_rate:.2f}%")
    
    relevancy_pass_rate = (sum(relevancy_results) / len(relevancy_results)) * 100
    print(f"Relevancy Pass Rate: {relevancy_pass_rate:.2f}%")
    
    print("="*60)

if __name__ == "__main__":
    main()
