from rag.retriever import retrieve
from rag.generator import build_query, generate_explanation

def explain(decision: str, features: dict) -> str:
    """End-to-end RAG pipeline: retrieves regulation text and generates explanation."""
    # 1. Build a query using the decision and applicant features
    query = build_query(decision, features)
    
    # 2. Retrieve relevant regulation chunks from ChromaDB
    chunks = retrieve(query, k=3)
    
    # 3. Generate explanation using Flan-T5
    explanation = generate_explanation(decision, features, chunks)
    
    return explanation
