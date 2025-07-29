import os
import pandas as pd
from dotenv import load_dotenv
from rag_core import initialize_rag_system, answer_with_rag
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.groq import Groq 

# Load environment variables from .env file
load_dotenv()

# Evaluation Dataset
EVALUATION_DATASET = [
    {
        "query": "دانشجوی مشروط در هر ترم حداکثر چند واحد میتواند اخذ کند؟",
        "reference_answer": "بر اساس ماده ۱۹ آیین‌نامه، دانشجویی که مشروط می‌شود، در نیمسال بعدی حداکثر می‌تواند تا ۱۴ واحد درسی انتخاب کند."
    },
    {
        "query": "مدت مجاز تحصیل در دوره کارشناسی پیوسته چقدر است؟ و چقدر میتوان آن را افزایش داد؟",
        "reference_answer": "بر اساس ماده ۱۵ آیین‌نامه، مدت مجاز تحصیل در دوره کارشناسی پیوسته چهار سال است. دانشگاه اختیار دارد در شرایط خاص این مدت را حداکثر دو نیمسال افزایش دهد."
    },
    {
        "query": "حداقل نمره قبولی برای هر درس چند است؟",
        "reference_answer": "مطابق ماده ۱۸ آیین‌نامه، حداقل نمره قبولی در هر درس ۱۰ است."
    },
    {
        "query": "تغییر رشته از دوره شبانه به روزانه امکان پذیر است؟",
        "reference_answer": "خیر، بر اساس ماده ۲۴، تغییر رشته از دوره شبانه به روزانه ممنوع است، اما برعکس آن مجاز است."
    }
]

def initialize_correctness_evaluator():
    """Initializes the CorrectnessEvaluator using the Groq LLM."""
    print("Initializing Correctness Evaluator with Groq...")
    
    eval_llm = Groq(
        model="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    correctness_evaluator = CorrectnessEvaluator(llm=eval_llm)
    print("Evaluator is ready.")
    return correctness_evaluator

def run_correctness_evaluation(dataset, evaluator):
    """
    Runs the evaluation by calling the actual RAG function from rag_core.
    """
    evaluation_results = []
    print(f"\nRunning evaluation on {len(dataset)} questions...")

    for i, item in enumerate(dataset):
        query = item["query"]
        reference_answer = item["reference_answer"]
        print(f"Processing Question {i+1}/{len(dataset)}: {query[:50]}...")

        # Get response using the *exact same function* as the web app
        generated_answer = answer_with_rag(query)
        
        # Run Correctness evaluation
        correctness_result = evaluator.evaluate(
            query=query,
            response=generated_answer,
            reference=reference_answer,
        )

        # Store the required results
        evaluation_results.append({
            "Query": query,
            "Reference Answer": reference_answer,
            "Generated Answer": generated_answer,
            "Correctness Score (1-5)": correctness_result.score,
            "Feedback": correctness_result.feedback,
        })
        
    print("\nEvaluation run complete.")
    return pd.DataFrame(evaluation_results)

def main():
    """Main function to execute the entire process."""
    
    # Initialize the RAG system once using the function from rag_core.py
    # This ensures all settings (LLM, retriever, prompt) are identical.
    initialize_rag_system()
    print("-" * 50)

    # Run Evaluation
    correctness_evaluator = initialize_correctness_evaluator()
    evaluation_df = run_correctness_evaluation(EVALUATION_DATASET, correctness_evaluator)

    # Save results to an Excel file
    output_filename = "evaluation_results.xlsx"
    evaluation_df.to_excel(output_filename, index=False, engine='openpyxl')
    
    print(f"✅ Results successfully saved to '{output_filename}'")

if __name__ == "__main__":
    main()