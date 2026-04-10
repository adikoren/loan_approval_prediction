from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def build_query(decision: str, features: dict) -> str:
    """Build a natural query from the model's decision + key features to query ChromaDB."""
    return (
        f"Loan {decision}. "
        f"Loan type code: {features.get('loan_type', 'unknown')}. "
        f"Applicant income: ${features.get('applicant_income', 'unknown')}. "
        f"Loan amount: ${features.get('loan_amount', 'unknown')}."
    )

def generate_explanation(decision: str, features: dict, chunks: list[str]) -> str:
    """Generate a natural language explanation grounded in regulations using Flan-T5."""
    context = "\n".join(chunks)
    
    prompt = f"Explain briefly why a loan application was {decision.upper()} based on the context.\nContext: {context}\nIncome: ${features.get('applicant_income')}, Loan amount: ${features.get('loan_amount')}, Loan type code: {features.get('loan_type')}\nExplanation:"

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, min_length=20)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
