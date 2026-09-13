import os
import re

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

# ingest.py never stored per-chunk metadata, but every chunk id already
# encodes its source PDF as "{filename}_{index}" — parse that instead of
# re-ingesting just to add a metadata field.
_CHUNK_ID_PATTERN = re.compile(r"^(?P<source>.+\.pdf)_\d+$")


def _source_of(chunk_id: str) -> str:
    match = _CHUNK_ID_PATTERN.match(chunk_id)
    return match.group("source") if match else chunk_id


# The FHA Handbook makes up ~1,433 of the corpus's 1,572 chunks (~91%), so
# unfiltered semantic search keeps surfacing FHA-specific underwriting/MIP
# rules for Conventional and VA queries simply because that source dominates
# the corpus — not because it's actually applicable regulatory guidance for
# those loan types.
FHA_SOURCE = "fha_handbook.pdf"


def _excluded_sources_for(loan_type: str | None) -> set[str]:
    """Sources that should not ground an explanation for this loan type."""
    if loan_type == "FHA-insured":
        return set()
    if loan_type in {"Conventional", "VA-guaranteed", "FSA/RHS-guaranteed"}:
        return {FHA_SOURCE}
    # Unknown/missing loan_type: no basis to exclude anything.
    return set()


def retrieve(query: str, k: int = 3, loan_type: str | None = None) -> list[dict]:
    """Retrieves up to k relevant chunks from ChromaDB for a given query.

    Excludes source documents that aren't applicable to the application's
    loan type (see _excluded_sources_for) — filtering happens after the
    similarity search, over a larger candidate pool, so results are never
    backfilled with an excluded source just to reach k; a loan type with no
    genuinely relevant chunks left can legitimately get fewer than k, or
    none.

    Returns each chunk as {"source": <pdf filename>, "chunk_id": ..., "text": ...}
    so callers can verify what actually grounded a generated explanation.
    """
    excluded = _excluded_sources_for(loan_type)

    query_vec = [vec.tolist() for vec in model.embed([query])]
    candidate_pool = min(collection.count(), max(k * 5, 15))
    results = collection.query(query_embeddings=query_vec, n_results=candidate_pool)

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    for chunk_id, text in zip(results["ids"][0], results["documents"][0]):
        source = _source_of(chunk_id)
        if source in excluded:
            continue
        chunks.append({"source": source, "chunk_id": chunk_id, "text": text})
        if len(chunks) == k:
            break

    return chunks
