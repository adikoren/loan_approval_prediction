import chromadb
import pdfplumber
from fastembed import TextEmbedding
import glob
import os

CHUNK_SIZE = 300    # words per chunk
OVERLAP = 50        # word overlap between chunks
# Same model as the original design (sentence-transformers/all-MiniLM-L6-v2,
# 384-dim), served through fastembed's ONNX runtime instead of full PyTorch.
# This keeps the documented embedding choice but avoids pulling in torch +
# ~3GB of CUDA runtime libraries for what is pure CPU inference in a
# container — see EXPLANATION.md for the original model rationale.
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Anchored to the project root so this always writes to the same path
# rag/retriever.py reads from, regardless of the process's cwd when this
# script is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "rag_db")
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
        
    model = TextEmbedding(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION)

    # ChromaDB requires string IDs
    collection.add(
        documents=chunks,
        embeddings=[vec.tolist() for vec in model.embed(chunks)],
        ids=[f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
    )
    print(f"Successfully ingested {len(chunks)} chunks from {pdf_path}")

if __name__ == "__main__":
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        print("No PDF files found in docs/")
    for pdf in pdf_files:
        ingest_pdf(pdf)
