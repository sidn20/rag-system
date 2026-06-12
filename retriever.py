from embedder import Embedder
from vector_store import VectorStore
from chunker import load_and_chunk_documents
import os

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
        """
        Retrieve relevant chunks and build an answer.
        In a full RAG system this context would be sent to an LLM.
        Here we return the context itself as the answer.
        """
        retrieved = self.retrieve(query, top_k=top_k)

        # Build context from retrieved chunks
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

if __name__ == "__main__":
    print("Initializing RAG system...\n")
    rag = RAGRetriever()

    test_queries = [
        "what is machine learning",
        "who created linux",
        "what are python libraries",
        "what is reinforcement learning"
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: {query}")
        print(f"{'='*55}")
        result = rag.answer(query)
        print(f"Sources: {result['sources']}")
        print(f"Chunks used: {result['chunks_used']}")
        print(f"\nContext retrieved:")
        for r in rag.retrieve(query, top_k=2):
            print(f"  [{r['source']} | score: {r['relevance_score']}]")
            print(f"  {r['text'][:100]}...")
