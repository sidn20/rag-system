FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
COPY chunker.py .
COPY embedder.py .
COPY vector_store.py .
COPY retriever.py .
COPY api.py .
COPY documents/ documents/
COPY vector_store.faiss .
COPY vector_store.meta .

EXPOSE 8000

CMD ["python3", "api.py"]
