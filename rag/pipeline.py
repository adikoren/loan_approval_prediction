from rag.retriever import retrieve
from rag.generator import build_query, generate_explanation


def explain(
    decision: str,
    features: dict,
    confidence: float | None = None,
    approval_probability: float | None = None,
) -> dict:
    """End-to-end RAG pipeline: retrieves regulation text and generates explanation.

    Returns {"explanation": str, "sources": [...]} — sources lists exactly
    which retrieved chunks grounded this explanation (document, issuer,
    document_type, loan_program, source_url, section), so grounding can be
    verified rather than assumed.
    """
    # 1. Build a query using the decision, confidence, and applicant features
    query = build_query(decision, features, confidence, approval_probability)

    # 2. Retrieve relevant regulation chunks from ChromaDB: program-specific
    #    underwriting chunks for this application's loan type, plus general
    #    compliance/fair-lending/reporting chunks (see rag/retriever.py)
    loan_type = features.get("loan_type")
    try:
        chunks = retrieve(query, k=3, loan_type=loan_type)
    except Exception as e:
        print(f"[rag.pipeline] Retrieval failed: {e}. Continuing without regulation context.")
        chunks = []

    print(
        f"[rag.pipeline] loan_type={loan_type!r} retrieved {len(chunks)} chunk(s): "
        + ", ".join(f"{c['chunk_id']} ({c['loan_program']}/{c['document_type']})" for c in chunks)
    )

    # 3. Generate explanation using Claude, grounded only in the retrieved chunks
    explanation = generate_explanation(decision, features, chunks, confidence, approval_probability)

    return {
        "explanation": explanation,
        "sources": [
            {
                "chunk_id": c["chunk_id"],
                "document": c["source"],
                "issuer": c["issuer"],
                "document_type": c["document_type"],
                "loan_program": c["loan_program"],
                "source_url": c["source_url"],
                "section": c["section"],
                "source_format": c["source_format"],
                "content_scope": c["content_scope"],
            }
            for c in chunks
        ],
    }
