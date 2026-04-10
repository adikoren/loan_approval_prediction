import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = 'all-MiniLM-L6-v2'
DB_PATH = "./rag_db"
COLLECTION = "loan_regulations"

# Initialize model and database collection
model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION)

def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieves top-k relevant chunks from ChromaDB for a given query."""
    query_vec = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=k)
    
    # Return empty list if no results
    if not results['documents'] or not results['documents'][0]:
        return []
        
    return results['documents'][0]
