# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course QA is a RAG-based Q&A assistant for course textbooks. Students ask natural language questions, the system retrieves relevant content from the textbook PDF and generates answers with citations.

Architecture: PDF → parse → chunk → embed → ChromaDB → retrieve Top-K → LLM → answer with source citations.

## Commands

```bash
# Activate environment
conda activate course-qa

# One-command start (auto-detect PDF, ingest if needed, launch web UI)
python start.py

# Manual ingestion
python ingest.py

# Start Streamlit UI
streamlit run app.py
```

## Key Technical Details

- **Environment**: conda `course-qa`, Python 3.10, PyTorch 2.6+cu124, GPU with 8GB+ VRAM
- **Embedding**: BGE-M3 local model, path configured in `config.yaml` (`embedding.model_path`).
- **Chapter detection**: Regex-based (`ingest.py:CHAPTER_PATTERNS`), supports Chinese (`第X章`) and English (`Chapter X`, `X.X`). Line-by-line scan.
- **Config**: All settings in `config.yaml` — chunk size, model paths, LLM params, retrieval params. `load_config()` is the single entry point.
- **LLM**: Any OpenAI-compatible API (DeepSeek / OpenAI / Ollama etc.), configured in `config.yaml` `llm` section.
- **API Key**: `config.yaml` 中的 `api_key` 字段仅用于本地开发，commit 前必须清空。
- **One-command start**: `start.py` scans `data/` for PDFs, checks `chroma_db/.ingested` marker to decide whether to re-ingest, then launches Streamlit.

## Development Workflow

Each phase follows this sequence: implement → verify → `/check` review → update README if docs diverge from implementation → git commit and push (only when user explicitly requests).

5-phase build plan: (1) infrastructure, (2) PDF parsing + chunking, (3) BGE-M3 embedding + ChromaDB, (4) RAG retrieval + LLM generation, (5) Streamlit UI + one-command start.
