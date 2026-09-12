#!/bin/sh
set -e

# The RAG vector store (rag_db/) is built from docs/*.pdf using an embedding
# model pulled from Hugging Face on first use. That download can't happen at
# `docker build` time in every environment (some CI/sandbox networks block
# huggingface.co), so it happens once here at container startup instead,
# where the runtime network is expected to be open. Skips the rebuild if
# rag_db/ is already populated (e.g. restored from a mounted volume).
if [ -z "$(ls -A /app/rag_db 2>/dev/null)" ]; then
  echo "[entrypoint] rag_db/ is empty — building the RAG vector store from docs/*.pdf..."
  python rag/ingest.py
else
  echo "[entrypoint] rag_db/ already populated — skipping ingest."
fi

echo "[entrypoint] Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
