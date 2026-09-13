from rag.retriever import retrieve
from rag.generator import build_query, generate_explanation


def explain(
    decision: str,
    features: dict,
    confidence: float | None = None,
    approval_probability: float | None = None,
) -> dict:
    """End-to-end RAG pipeline: retrieves regulation text and generates explanation.

    Returns {"explanation": str, "sources": [{"source": ..., "chunk_id": ...}, ...]}
    — sources lists exactly which retrieved chunks grounded this explanation,
    so grounding can be verified rather than assumed.
    """
    # 1. Build a query using the decision, confidence, and applicant features
    query = build_query(decision, features, confidence, approval_probability)

    # 2. Retrieve relevant regulation chunks from ChromaDB, restricted to
    #    sources applicable to this application's loan type
    loan_type = features.get("loan_type")
    try:
        chunks = retrieve(query, k=3, loan_type=loan_type)
    except Exception as e:
        print(f"[rag.pipeline] Retrieval failed: {e}. Continuing without regulation context.")
        chunks = []

    print(
        f"[rag.pipeline] loan_type={loan_type!r} retrieved {len(chunks)} chunk(s): "
        f"{[c['chunk_id'] for c in chunks]}"
    )

    # 3. Generate explanation using Claude, grounded in the retrieved chunks
    explanation = generate_explanation(decision, features, chunks, confidence, approval_probability)

    return {
        "explanation": explanation,
        "sources": [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks],
    }
