import os
import subprocess
import time
from embedder import Embedder
from vector_store import VectorStore
from chunker import load_and_chunk_documents

class RAGRetriever:
    def __init__(self, docs_folder: str = "documents", store_path: str = "vector_store"):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.store_path = store_path

        if os.path.exists(store_path + ".faiss"):
            print("Loading existing vector store...")
            self.store.load(store_path)
        else:
            print("Building vector store from documents...")
            self._build_store(docs_folder)

    def _build_store(self, docs_folder: str):
        chunks = load_and_chunk_documents(docs_folder)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)
        self.store.save(self.store_path)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Find most relevant chunks for a query."""
        query_vector = self.embedder.embed_text(query)
        results = self.store.search(query_vector, top_k=top_k)

        retrieved = []
        for dist, ec in results:
            retrieved.append({
                "text": ec.chunk.text,
                "source": ec.chunk.source,
                "distance": round(dist, 4),
                "relevance_score": round(1 / (1 + dist), 4)
            })
        return retrieved

    def answer(self, query: str, top_k: int = 3) -> dict:
        """Retrieve relevant chunks and return context as answer."""
        retrieved = self.retrieve(query, top_k=top_k)

        context = "\n\n".join([
            f"[From {r['source']} | relevance: {r['relevance_score']}]\n{r['text']}"
            for r in retrieved
        ])

        return {
            "query": query,
            "context": context,
            "sources": list(set(r["source"] for r in retrieved)),
            "chunks_used": len(retrieved),
            "answer": f"Based on the documents:\n\n{context}"
        }

    def answer_with_llm(self, query: str, top_k: int = 3) -> dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Build context
        3. Send to local LLM via Ollama
        4. Return generated answer
        """
        # Step 1 — Retrieve relevant chunks
        retrieved = self.retrieve(query, top_k=top_k)

        # Step 2 — Build context from chunks
        context = "\n\n".join([
            f"[Source: {r['source']} | relevance: {r['relevance_score']}]\n{r['text']}"
            for r in retrieved
        ])

        # Step 3 — Build prompt
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

        # Step 4 — Send to Ollama
        start = time.perf_counter()
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True
        )
        llm_latency = (time.perf_counter() - start) * 1000
        answer = result.stdout.strip()

        return {
            "query": query,
            "answer": answer,
            "sources": list(set(r["source"] for r in retrieved)),
            "chunks_used": len(retrieved),
            "context": context,
            "llm_latency_ms": round(llm_latency, 2)
        }


if __name__ == "__main__":
    print("Initializing RAG system with LLM...\n")
    rag = RAGRetriever()

    test_queries = [
        "what is machine learning",
        "who created linux",
        "what are python libraries",
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: {query}")
        print(f"{'='*55}")
        result = rag.answer_with_llm(query)
        print(f"Sources  : {result['sources']}")
        print(f"Latency  : {result['llm_latency_ms']:.0f}ms")
        print(f"\nAnswer:\n{result['answer']}")
