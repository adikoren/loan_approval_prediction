from rag.retriever import retrieve
from rag.generator import build_query, generate_explanation

def explain(decision: str, features: dict, confidence: float | None = None) -> str:
    """End-to-end RAG pipeline: retrieves regulation text and generates explanation."""
    # 1. Build a query using the decision, confidence, and applicant features
    query = build_query(decision, features, confidence)

    # 2. Retrieve relevant regulation chunks from ChromaDB
    try:
        chunks = retrieve(query, k=3)
    except Exception as e:
        print(f"[rag.pipeline] Retrieval failed: {e}. Continuing without regulation context.")
        chunks = []

    # 3. Generate explanation using Claude, grounded in the retrieved chunks
    explanation = generate_explanation(decision, features, chunks, confidence)

    return explanation
