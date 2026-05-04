"""One-command startup: detect PDF, ingest if needed, launch Streamlit."""

import json
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data")
MARKER_FILE = Path("chroma_db/.ingested")


def find_pdfs():
    """Scan data/ directory for PDF files."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)


def get_marker():
    """Read the ingestion marker file."""
    if MARKER_FILE.exists():
        with open(MARKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_marker(pdf_path):
    """Save ingestion marker."""
    MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    stat = pdf_path.stat()
    with open(MARKER_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "path": str(pdf_path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }, f)


def needs_ingestion(pdf_path):
    """Check if PDF needs (re-)ingestion."""
    marker = get_marker()
    stat = pdf_path.stat()
    return (
        marker.get("path") != str(pdf_path)
        or marker.get("size") != stat.st_size
        or marker.get("mtime") != stat.st_mtime
    )


def main():
    print("=" * 50)
    print("  Course QA — 课程教材问答助手")
    print("=" * 50)

    # Step 1: Find PDF
    pdfs = find_pdfs()
    if not pdfs:
        print("\n[!] data/ 目录下没有找到 PDF 文件。")
        print("    请将教材 PDF 放入 data/ 目录后重新运行。")
        sys.exit(1)

    pdf_path = pdfs[0]
    if len(pdfs) > 1:
        print(f"\n检测到 {len(pdfs)} 个 PDF，使用最新的: {pdf_path.name}")
        for p in pdfs:
            print(f"  - {p.name} ({'选用' if p == pdf_path else ''})")
    else:
        print(f"\n检测到 PDF: {pdf_path.name}")

    # Step 2: Ingest if needed
    if needs_ingestion(pdf_path):
        print("\n检测到新 PDF 或文件变更，开始解析...")
        from ingest import main as ingest_main
        ingest_main(pdf_path=str(pdf_path))
        save_marker(pdf_path)
        print("解析完成！")
    else:
        print("\nPDF 未变更，跳过解析。")

    # Step 3: Launch Streamlit
    print("\n启动问答界面...")
    print("浏览器将自动打开 http://localhost:8501")
    print("按 Ctrl+C 停止服务。\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
