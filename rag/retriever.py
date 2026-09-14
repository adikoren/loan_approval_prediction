import os

import chromadb
from fastembed import TextEmbedding

# Same model as ingest.py — served via fastembed's ONNX runtime (see
# ingest.py for why this replaced sentence-transformers/torch).
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Anchored to the project root (parent of this package) rather than a
# cwd-relative "./rag_db" — a relative path silently resolves to whatever
# directory the process happened to be launched from, and chromadb's
# get_or_create_collection() won't error on a wrong/empty path, it just
# returns an empty collection, so retrieval silently returns zero chunks
# instead of failing loudly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "rag_db")
COLLECTION = "loan_regulations"

# Initialize model and database collection
model = TextEmbedding(model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION)

# Maps the application's loan_type value to the loan_program metadata value
# attached to underwriting chunks at ingest time (see rag/ingest.py's
# DOCUMENTS manifest). An unmapped/missing loan_type has no program-specific
# underwriting source to draw on, so retrieval falls back to general
# compliance material only — it never guesses a program.
LOAN_TYPE_TO_PROGRAM = {
    "Conventional": "conventional",
    "FHA-insured": "fha",
    "VA-guaranteed": "va",
    "FSA/RHS-guaranteed": "usda",
}


def _program_for(loan_type: str | None) -> str | None:
    return LOAN_TYPE_TO_PROGRAM.get(loan_type)


def _run_query(query_vec, where: dict, n_results: int) -> list[dict]:
    if n_results <= 0:
        return []
    results = collection.query(query_embeddings=query_vec, n_results=n_results, where=where)
    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for chunk_id, text, meta, distance in zip(results["ids"][0], results["documents"][0], metadatas, distances):
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "distance": distance,
            "source": meta.get("document"),
            "issuer": meta.get("issuer"),
            "loan_program": meta.get("loan_program"),
            "document_type": meta.get("document_type"),
            "source_url": meta.get("source_url"),
            "section": meta.get("section") or None,
            "source_format": meta.get("source_format"),
            "content_scope": meta.get("content_scope"),
        })
    return chunks


def retrieve(query: str, k: int = 3, loan_type: str | None = None) -> list[dict]:
    """Retrieves up to k relevant chunks, split across two independent,
    metadata-filtered ChromaDB queries rather than one unfiltered pool:

    1. Program-specific underwriting chunks (loan_program == the program
       mapped from loan_type, document_type == "underwriting"). A loan type
       with no ingested program-specific document (e.g. no VA source exists)
       or an unmapped/missing loan_type simply gets none — retrieval never
       falls back to a different program's underwriting material just
       because it scores well semantically.
    2. General compliance/fair-lending/reporting chunks (loan_program ==
       "general"), which apply regardless of program and backfill any
       leftover slots so program-specific scarcity doesn't starve the
       explanation of all grounding.

    Returns each chunk with its full source metadata (document, issuer,
    document_type, loan_program, source_url, section) so both the generator
    and the API response can show exactly what grounded the explanation.
    """
    program = _program_for(loan_type)
    query_vec = [vec.tolist() for vec in model.embed([query])]

    # With a known program, reserve most slots for program-specific
    # underwriting and at least one for general context; with none, general
    # material is all there is to retrieve.
    general_slots = 1 if program else k
    program_slots = k - general_slots if program else 0

    program_chunks = []
    if program and program_slots > 0:
        where = {"$and": [{"loan_program": program}, {"document_type": "underwriting"}]}
        candidate_pool = min(collection.count(), max(program_slots * 5, 15))
        program_chunks = _run_query(query_vec, where, candidate_pool)[:program_slots]

    # Backfill any unused program slots (including all of them, if the
    # program has no ingested underwriting source) with general material,
    # so a program gap degrades to "less specific" rather than "no grounding
    # at all" — without ever substituting another program's underwriting.
    general_slots += max(0, program_slots - len(program_chunks))

    # Within "general", reserve at least one slot for fair-lending material
    # (ECOA/Regulation B) before backfilling with reporting material (HMDA).
    # HMDA is ~15x larger than the ECOA corpus, so an unweighted query over
    # all of "general" nearly always drowns ECOA out — every result comes
    # back as HMDA's data-field/edit-check tables, which are reporting
    # mechanics, not the fair-lending framing this explanation actually
    # needs (see rag/generator.py's SYSTEM_PROMPT).
    fair_lending_slots = min(1, general_slots) if general_slots > 0 else 0
    candidate_pool = min(collection.count(), max(fair_lending_slots * 5, 15))
    fair_lending_chunks = _run_query(
        query_vec, {"$and": [{"loan_program": "general"}, {"document_type": "fair_lending"}]}, candidate_pool
    )[:fair_lending_slots]

    reporting_slots = general_slots - len(fair_lending_chunks)
    candidate_pool = min(collection.count(), max(reporting_slots * 5, 15))
    reporting_chunks = _run_query(
        query_vec, {"$and": [{"loan_program": "general"}, {"document_type": "reporting"}]}, candidate_pool
    )[:reporting_slots]

    return program_chunks + fair_lending_chunks + reporting_chunks
