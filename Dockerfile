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

# Model artifacts — preprocessor.joblib bakes all encoding maps from train.csv
# at image build time (~145 KB), eliminating the need to ship train.csv (~500 MB)
# or load it at runtime.  This keeps container memory well below 512 MB.
COPY experiments/model.joblib ./experiments/model.joblib
COPY experiments/preprocessor.joblib ./experiments/preprocessor.joblib

# Pre-populated ChromaDB vector store so the container starts instantly
# without running ingest or downloading models at container boot time
COPY rag_db.tar.gz .
RUN tar -xzf rag_db.tar.gz && rm rag_db.tar.gz

# Pre-download the fastembed embedding model at build time
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{p}/health', timeout=3)" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
