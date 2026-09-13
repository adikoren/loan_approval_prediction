FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Application code
COPY config.py .
COPY src ./src
COPY rag ./rag
COPY app ./app
COPY frontend ./frontend
COPY docs ./docs
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Model artifact + the training data it needs to align inference requests
# against (see src/predict.py — target-mean encodings are learned from
# train.csv at request time, matching the src/predict.py batch contract).
COPY experiments/model.joblib ./experiments/model.joblib
COPY data/train.csv.gz ./data/train.csv.gz
RUN gzip -d ./data/train.csv.gz

# Pre-populated ChromaDB vector store so the container starts instantly
# without running ingest or downloading models at container boot time
COPY rag_db.tar.gz .
RUN tar -xzf rag_db.tar.gz && rm rag_db.tar.gz

# Pre-download the fastembed embedding model at build time
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
