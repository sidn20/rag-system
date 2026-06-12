import numpy as np
from sentence_transformers import SentenceTransformer
from chunker import Chunk
from dataclasses import dataclass

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: np.ndarray

# Using a small but powerful model — only 90MB
MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")

    def embed_text(self, text: str) -> np.ndarray:
        """Convert a single string to a vector."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Convert all chunks to vectors."""
        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]

        # encode all at once — much faster than one by one
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        embedded = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded.append(EmbeddedChunk(
                chunk=chunk,
                embedding=embedding.astype(np.float32)
            ))

        print(f"Done. Each embedding has {embeddings.shape[1]} dimensions.")
        return embedded

if __name__ == "__main__":
    from chunker import load_and_chunk_documents

    chunks = load_and_chunk_documents("documents")
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    print(f"\nSample embedding:")
    print(f"  Text: {embedded_chunks[0].chunk.text[:60]}...")
    print(f"  Vector shape: {embedded_chunks[0].embedding.shape}")
    print(f"  First 5 values: {embedded_chunks[0].embedding[:5]}")

