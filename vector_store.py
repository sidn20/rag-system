import numpy as np
import faiss
import pickle
from embedder import EmbeddedChunk

class VectorStore:
    def __init__(self, dimension: int = 384):
        # Flat L2 index — exact search, no approximation
        # For larger datasets you'd use IndexIVFFlat for speed
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[EmbeddedChunk] = []
        self.dimension = dimension

    def add(self, embedded_chunks: list[EmbeddedChunk]):
        """Add embedded chunks to the vector store."""
        vectors = np.array([ec.embedding for ec in embedded_chunks])
        self.index.add(vectors)
        self.chunks.extend(embedded_chunks)
        print(f"Added {len(embedded_chunks)} vectors. Total: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[tuple]:
        """
        Find top_k most similar chunks to the query vector.
        Returns list of (distance, EmbeddedChunk) tuples.
        Lower distance = more similar.
        """
        query = query_vector.reshape(1, -1)
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((float(dist), self.chunks[idx]))

        return results

    def save(self, path: str = "vector_store.pkl"):
        """Save the entire store to disk."""
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Vector store saved to {path}")

    def load(self, path: str = "vector_store.pkl"):
        """Load store from disk — no need to re-embed documents."""
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {self.index.ntotal} vectors from {path}")

if __name__ == "__main__":
    from chunker import load_and_chunk_documents
    from embedder import Embedder

    # Build the store
    chunks = load_and_chunk_documents("documents")
    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.add(embedded)

    # Test a search
    print("\nTesting search...")
    query = "what is machine learning"
    query_vector = embedder.embed_text(query)
    results = store.search(query_vector, top_k=2)

    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} results:\n")
    for dist, ec in results:
        print(f"  Distance: {dist:.4f}")
        print(f"  Source:   {ec.chunk.source}")
        print(f"  Text:     {ec.chunk.text[:100]}...")
        print()

    # Save for reuse
    store.save("vector_store")

