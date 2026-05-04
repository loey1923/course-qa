"""PDF ingestion pipeline: parse, chunk, and store in ChromaDB."""

import re
import sys
from pathlib import Path

import fitz
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def main():
    config = load_config()
    pdf_path = config["ingestion"]["pdf_path"]

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Please place your textbook PDF in the data/ directory.")
        sys.exit(1)

    print(f"Parsing PDF: {pdf_path}")
    pages = parse_pdf(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    print("Detecting chapters...")
    chapters = build_chapters(pages)
    print(f"  Found {len(chapters)} chapters/sections")
    for ch in chapters:
        print(f"    - {ch['title']} (p.{ch['start_page']}-{ch.get('end_page', '?')})")

    print("Chunking text...")
    chunks = chunk_chapters(chapters, config)
    print(f"  Generated {len(chunks)} chunks")

    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk["text"][:100].replace("\n", " ")
        print(f"  [{i}] ({chunk['metadata']['chapter']}) {preview}...")

    print(f"\nPhase 2 complete. {len(chunks)} chunks ready for embedding.")
    return chunks


if __name__ == "__main__":
    main()
