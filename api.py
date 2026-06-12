import time
import uuid
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import RAGRetriever
from datetime import datetime

app = FastAPI(
    title="RAG System API",
    description="Retrieval Augmented Generation API built from scratch",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    latency_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    avg_latency_ms: float

# Global state
rag = None
start_time = time.time()
request_log = []

@app.on_event("startup")
async def startup():
    global rag
    print("Loading RAG system...")
    rag = RAGRetriever()
    print("RAG system ready.")

@app.get("/health", response_model=HealthResponse)
def health():
    avg_latency = sum(r["latency_ms"] for r in request_log) / len(request_log) if request_log else 0
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - start_time, 2),
        total_requests=len(request_log),
        avg_latency_ms=round(avg_latency, 2)
    )

@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.perf_counter()
    result = rag.answer(request.question, top_k=request.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    request_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    request_log.append({
        "request_id": request_id,
        "question": request.question,
        "latency_ms": round(latency_ms, 2),
        "sources": result["sources"],
        "timestamp": timestamp,
        "mode": "retrieval"
    })

    with open("request_log.jsonl", "a") as f:
        f.write(json.dumps(request_log[-1]) + "\n")

    return QueryResponse(
        request_id=request_id,
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        latency_ms=round(latency_ms, 2),
        timestamp=timestamp
    )

@app.post("/predict/llm")
def predict_with_llm(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.perf_counter()
    result = rag.answer_with_llm(request.question, top_k=request.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    request_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    request_log.append({
        "request_id": request_id,
        "question": request.question,
        "latency_ms": round(latency_ms, 2),
        "sources": result["sources"],
        "timestamp": timestamp,
        "mode": "llm"
    })

    with open("request_log.jsonl", "a") as f:
        f.write(json.dumps(request_log[-1]) + "\n")

    return {
        "request_id": request_id,
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "llm_latency_ms": result["llm_latency_ms"],
        "total_latency_ms": round(latency_ms, 2),
        "timestamp": timestamp
    }

@app.get("/metrics")
def metrics():
    if not request_log:
        return {"message": "No requests yet"}
    latencies = [r["latency_ms"] for r in request_log]
    llm_requests = [r for r in request_log if r.get("mode") == "llm"]
    retrieval_requests = [r for r in request_log if r.get("mode") == "retrieval"]
    return {
        "total_requests": len(request_log),
        "llm_requests": len(llm_requests),
        "retrieval_requests": len(retrieval_requests),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "recent_requests": request_log[-5:]
    }

@app.get("/sources")
def sources():
    import os
    docs = os.listdir("documents")
    return {"documents_loaded": docs, "total": len(docs)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
