"""PDF ingestion pipeline: parse, chunk, embed, and store in ChromaDB."""

import re
import sys
from pathlib import Path

import chromadb
import fitz
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CHAPTER_PATTERNS = [
    # Chinese: 第1章, 第一章, 第10章
    re.compile(r"^第[一二三四五六七八九十百千\d]+[章节篇]"),
    # English: Chapter 1, CHAPTER 1
    re.compile(r"^chapter\s+\d+", re.IGNORECASE),
    # Numbered sections: 1.1, 2.3.1
    re.compile(r"^\d+\.\d+"),
    # Part / Unit / Lecture
    re.compile(r"^(part|unit|lecture)\s+\d+", re.IGNORECASE),
]


def is_chapter_heading(text):
    """Check if a line looks like a chapter/section heading."""
    stripped = text.strip()
    if not stripped or len(stripped) > 80:
        return False
    for pattern in CHAPTER_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def parse_pdf(pdf_path):
    """Parse PDF into pages with text and metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page_num": page_num + 1,
                "text": text.strip(),
            })
    doc.close()
    return pages


def build_chapters(pages):
    """Group pages into chapter-level blocks based on heading detection."""
    chapters = []
    current = {"title": "前言", "start_page": 1, "text": ""}

    for page in pages:
        for line in page["text"].split("\n"):
            if is_chapter_heading(line):
                if current["text"].strip():
                    current["end_page"] = page["page_num"]
                    chapters.append(current)
                current = {
                    "title": line.strip(),
                    "start_page": page["page_num"],
                    "text": "",
                }
            else:
                current["text"] += "\n" + line

    if current["text"].strip():
        current["end_page"] = pages[-1]["page_num"] if pages else 1
        chapters.append(current)

    return chapters


def chunk_chapters(chapters, config):
    """Split chapters into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["ingestion"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunk_overlap"],
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    all_chunks = []
    for chapter in chapters:
        chunks = splitter.split_text(chapter["text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chapter": chapter["title"],
                    "page_start": chapter["start_page"],
                    "page_end": chapter.get("end_page", chapter["start_page"]),
                    "chunk_index": i,
                },
            })

    return all_chunks


def load_embedding_model(config):
    """Load BGE-M3 embedding model."""
    model_path = config["embedding"]["model_path"]
    device = config["embedding"]["device"]
    print(f"Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path, device=device)
    print(f"  Model loaded on {device}")
    return model


def embed_chunks(model, chunks, config):
    """Generate embeddings for all chunks."""
    texts = [c["text"] for c in chunks]
    batch_size = config["embedding"]["batch_size"]
    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    return embeddings


def store_in_chromadb(chunks, embeddings, config):
    """Store chunks and embeddings in ChromaDB."""
    persist_dir = config["vectorstore"]["persist_dir"]
    collection_name = config["vectorstore"]["collection_name"]

    print(f"Storing in ChromaDB: {persist_dir}/{collection_name}")
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if present
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"  Stored {collection.count()} chunks")
    return collection


def main(pdf_path=None):
    config = load_config()
    if pdf_path is None:
        pdf_path = config["ingestion"]["pdf_path"]

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Please place your textbook PDF in the data/ directory.")
        sys.exit(1)

    # Step 1: Parse PDF
    print(f"Parsing PDF: {pdf_path}")
    pages = parse_pdf(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    # Step 2: Detect chapters
    print("Detecting chapters...")
    chapters = build_chapters(pages)
    print(f"  Found {len(chapters)} chapters/sections")
    for ch in chapters:
        print(f"    - {ch['title']} (p.{ch['start_page']}-{ch.get('end_page', '?')})")

    # Step 3: Chunk
    print("Chunking text...")
    chunks = chunk_chapters(chapters, config)
    print(f"  Generated {len(chunks)} chunks")

    # Step 4: Embed
    model = load_embedding_model(config)
    embeddings = embed_chunks(model, chunks, config)

    # Step 5: Store
    collection = store_in_chromadb(chunks, embeddings, config)

    # Step 6: Verify
    print("\nVerifying retrieval...")
    test_query = chunks[0]["text"][:50]
    query_embedding = model.encode([test_query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    print(f"  Query: '{test_query}...'")
    for i, doc in enumerate(results["documents"][0]):
        dist = results["distances"][0][i]
        meta = results["metadatas"][0][i]
        print(f"  [{i}] dist={dist:.4f} ({meta['chapter']}, p.{meta['page_start']}) {doc[:60]}...")

    print(f"\nPhase 3 complete. {collection.count()} chunks stored in ChromaDB.")


if __name__ == "__main__":
    main()
