import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer
import glob
import os

CHUNK_SIZE = 300    # words per chunk
OVERLAP = 50        # word overlap between chunks
EMBED_MODEL = 'all-MiniLM-L6-v2'
DB_PATH = "./rag_db"
COLLECTION = "loan_regulations"

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+CHUNK_SIZE]))
        i += CHUNK_SIZE - OVERLAP
    return chunks

def ingest_pdf(pdf_path: str):
    print(f"Ingesting {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    chunks = chunk_text(text)
    
    # Ensure there are chunks to add
    if not chunks:
        print(f"Warning: No text extracted from {pdf_path}")
        return
        
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION)
    
    # ChromaDB requires string IDs
    collection.add(
        documents=chunks,
        embeddings=model.encode(chunks).tolist(),
        ids=[f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
    )
    print(f"Successfully ingested {len(chunks)} chunks from {pdf_path}")

if __name__ == "__main__":
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        print("No PDF files found in docs/")
    for pdf in pdf_files:
        ingest_pdf(pdf)
