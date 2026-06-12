# RAG System — Built from Scratch

A Retrieval Augmented Generation (RAG) system built from scratch in Python
without LangChain or LlamaIndex. Includes a FastAPI REST API, vector search,
semantic embeddings, and concurrent load testing.

## What is RAG?

Normal AI answers from training data only. RAG finds relevant documents
first, then uses them to answer — enabling AI to answer from YOUR data.
Question → Embed Query → Vector Search → Retrieve Chunks → Answer

## Architecture
documents/
↓
chunker.py       — splits documents into overlapping chunks
↓
embedder.py      — converts chunks to 384-dim vectors (all-MiniLM-L6-v2)
↓
vector_store.py  — stores and searches vectors using FAISS
↓
retriever.py     — finds relevant chunks for any query
↓
api.py           — FastAPI REST API serving predictions
↓
load_tester.py   — async concurrent load testing

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status and uptime |
| `/predict` | POST | Fast retrieval-only answer (62ms) |
| `/predict/llm` | POST | Full RAG answer via Llama3.2 (6-10s) |
| `/metrics` | GET | Latency stats, request history |
| `/sources` | GET | List loaded documents |

## Sample LLM Response

```json
{
  "request_id": "ba8e3959",
  "question": "who created linux and when",
  "answer": "Linus Torvalds in 1991.",
  "sources": ["linux_basics.txt"],
  "chunks_used": 2,
  "llm_latency_ms": 6853.76,
  "total_latency_ms": 6888.42
}
```
## Performance

Tested with 20 concurrent requests:

| Metric | Value |
|--------|-------|
| Success rate | 100% (20/20) |
| Mean latency | 126ms |
| Min latency | 86ms |
| Max latency | 155ms |
| Stdev | 19ms |

## Quick Start

```bash
# Install dependencies
pip3 install sentence-transformers faiss-cpu fastapi uvicorn aiohttp

# Start the API
python3 api.py

# Ask a question
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "what is machine learning", "top_k": 2}'

# Run load test
python3 load_tester.py
```

## Sample Response

```json
{
  "request_id": "479bb060",
  "question": "what is machine learning",
  "answer": "Based on the documents: [From ml_basics.txt | relevance: 0.7368] ...",
  "sources": ["ml_basics.txt"],
  "chunks_used": 2,
  "latency_ms": 62.24,
  "timestamp": "2026-06-12 06:53:38"
}
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Local text embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Facebook's vector similarity search |
| `FastAPI` | REST API framework |
| `uvicorn` | ASGI server |
| `aiohttp` + `asyncio` | Async concurrent load testing |
| `pydantic` | Request/response validation |

## Project Structure

rag_system/
├── chunker.py        — document chunking with overlap
├── embedder.py       — text to vector conversion
├── vector_store.py   — FAISS vector storage and search
├── retriever.py      — RAG retrieval pipeline
├── api.py            — FastAPI REST API
├── load_tester.py    — async load testing
└── documents/        — source documents
├── ml_basics.txt
├── linux_basics.txt
└── python_basics.txt

## What I Learned

- How RAG works internally — chunking, embedding, retrieval
- FAISS vector indexing and similarity search
- Building production REST APIs with FastAPI and Pydantic
- Async concurrent programming with asyncio and aiohttp
- Semantic search vs keyword search
- Load testing and API performance measurement
