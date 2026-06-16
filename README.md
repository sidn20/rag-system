# RAG System — Built from Scratch

A fully containerized Retrieval Augmented Generation (RAG) system built
from scratch in Python without LangChain or LlamaIndex. Features dynamic
document ingestion, semantic search, local LLM inference, and a REST API.

## What is RAG?

Normal AI answers from training data only. RAG finds relevant documents
first, then uses them to answer — enabling AI to answer from YOUR data.

Question → Embed Query → FAISS Search → Retrieve Chunks → LLM Answer

## Architecture

documents/
↓
chunker.py       — splits documents into overlapping chunks
↓
embedder.py      — converts chunks to 384-dim vectors (all-MiniLM-L6-v2)
↓
vector_store.py  — stores and searches vectors using FAISS
↓
retriever.py     — RAG retrieval + Ollama LLM integration
↓
api.py           — FastAPI REST API with dynamic document ingestion
↓
load_tester.py   — async concurrent load testing

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status and uptime |
| `/predict` | POST | Fast retrieval-only answer (~54ms) |
| `/predict/llm` | POST | Full RAG answer via Llama3.2 (~7s) |
| `/upload` | POST | Upload and index new document instantly |
| `/metrics` | GET | Latency stats and request history |
| `/sources` | GET | List all indexed documents |

## Key Feature — Dynamic Document Ingestion

Upload any document at runtime — no code changes, no container rebuild:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.txt"
```

Response:
```json
{
  "message": "Document indexed successfully",
  "filename": "docker_basics.txt",
  "chunks_created": 3,
  "total_vectors": 9
}
```

The document is instantly searchable:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "what is docker", "top_k": 2}'
```

## Performance

Load tested with 20 concurrent requests:

| Metric | Value |
|--------|-------|
| Success rate | 100% (20/20) |
| Mean latency | 126ms |
| Min latency | 87ms |
| Max latency | 155ms |
| Retrieval only | ~54ms |
| With Llama3.2 | ~7s |

## Docker

```bash
# Build
docker build -t rag-system:v1 .

# Run
docker run -p 8000:8000 rag-system:v1

# Test
curl http://localhost:8000/health
```

## Quick Start (without Docker)

```bash
# Install dependencies
pip3 install fastapi uvicorn faiss-cpu sentence-transformers \
             aiohttp python-multipart

# Install Ollama for LLM support
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Start the API
python3 api.py

# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@documents/ml_basics.txt"

# Ask a question
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "what is machine learning", "top_k": 2}'

# Run load test
python3 load_tester.py
```

## Sample Responses

### Retrieval only (/predict)
```json
{
  "request_id": "2ac766d0",
  "question": "what is machine learning",
  "answer": "Based on the documents: [From ml_basics.txt | relevance: 0.7368] ...",
  "sources": ["ml_basics.txt"],
  "chunks_used": 2,
  "latency_ms": 54.22,
  "timestamp": "2026-06-12 06:53:38"
}
```

### Full LLM answer (/predict/llm)
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

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Local text embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Facebook's vector similarity search |
| `FastAPI` | REST API framework |
| `uvicorn` | ASGI server |
| `Ollama + Llama3.2` | Local LLM inference, no API key needed |
| `aiohttp + asyncio` | Async concurrent load testing |
| `pydantic` | Request/response validation |
| `Docker` | Containerization |

## Project Structure
rag_system/
├── chunker.py        — document chunking with overlap
├── embedder.py       — text to vector conversion
├── vector_store.py   — FAISS vector storage and search
├── retriever.py      — RAG pipeline + LLM integration
├── api.py            — FastAPI REST API
├── load_tester.py    — async load testing
├── Dockerfile        — container definition
├── requirements.txt  — Python dependencies
└── documents/        — source documents
├── ml_basics.txt
├── linux_basics.txt
├── python_basics.txt
└── docker_basics.txt

## What I Learned

- How RAG works internally — chunking, embedding, vector search, generation
- FAISS vector indexing and similarity search from scratch
- Building production REST APIs with FastAPI and Pydantic
- Local LLM inference with Ollama — no API keys or cloud needed
- Async concurrent programming with asyncio and aiohttp
- Dynamic document ingestion without system downtime
- Docker containerization for portable deployment
- Load testing and API performance measurement
