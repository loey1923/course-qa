# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course QA is a RAG-based Q&A assistant for course textbooks. Students ask natural language questions, the system retrieves relevant content from the textbook PDF and generates answers with citations.

Architecture: PDF → parse → chunk → embed → ChromaDB → retrieve Top-K → DeepSeek LLM → answer with source citations.

## Commands

```bash
# Activate environment
conda activate course-qa

# Run PDF ingestion pipeline
python ingest.py

# Start Streamlit UI
streamlit run app.py

# Install dependencies (after conda env creation)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Key Technical Details

- **Environment**: conda `course-qa`, Python 3.10, PyTorch 2.6+cu124, RTX 4060 8GB
- **HF cache**: All HuggingFace models cached at `E:/huggingface` (HF_HOME). BGE-M3 model at `E:/models/BGE-M3`.
- **Chapter detection**: Regex-based (`ingest.py:CHAPTER_PATTERNS`), supports Chinese (`第X章`) and English (`Chapter X`, `X.X`). Line-by-line scan, no `break` after first heading found.
- **Config**: All settings in `config.yaml` — chunk size, model paths, LLM params, retrieval params. `load_config()` is the single entry point.
- **LLM**: DeepSeek API via OpenAI-compatible client (`base_url: https://api.deepseek.com`).

## Development Workflow

Each Phase follows this sequence: implement → verify → `/check` review → update README if docs diverge from implementation → git commit and push.

5-phase build plan: (1) infrastructure, (2) PDF parsing + chunking, (3) BGE-M3 embedding + ChromaDB, (4) RAG retrieval + DeepSeek generation, (5) Streamlit UI.
