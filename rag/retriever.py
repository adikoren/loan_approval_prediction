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

def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieves top-k relevant chunks from ChromaDB for a given query."""
    query_vec = [vec.tolist() for vec in model.embed([query])]
    results = collection.query(query_embeddings=query_vec, n_results=k)
    
    # Return empty list if no results
    if not results['documents'] or not results['documents'][0]:
        return []
        
    return results['documents'][0]
