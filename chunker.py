from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int

def chunk_text(text: str, source: str, chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    overlap means consecutive chunks share some text —
    this prevents losing context at chunk boundaries.
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # Don't cut in the middle of a word — extend to next space
        if end < len(text):
            while end < len(text) and text[end] != ' ':
                end += 1

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append(Chunk(
                text=chunk_text_content,
                source=source,
                chunk_id=chunk_id,
                start_char=start,
                end_char=end
            ))
            chunk_id += 1

        # Move forward by chunk_size minus overlap
        # so consecutive chunks share 'overlap' characters
        start += chunk_size - overlap

    return chunks

def load_and_chunk_documents(docs_folder: str, chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    import os
    all_chunks = []

    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_folder, filename)
            with open(filepath, "r") as f:
                text = f.read()
            chunks = chunk_text(text, source=filename, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
            print(f"  {filename} → {len(chunks)} chunks")

    return all_chunks

if __name__ == "__main__":
    print("Loading and chunking documents...\n")
    chunks = load_and_chunk_documents("documents")
    print(f"\nTotal chunks: {len(chunks)}")
    print("\nSample chunks:")
    for c in chunks[:3]:
        print(f"\n[Chunk {c.chunk_id} from {c.source}]")
        print(f"  chars {c.start_char}-{c.end_char}")
        print(f"  text: {c.text[:80]}...")
