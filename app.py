"""Streamlit frontend for Course QA."""

import chromadb
import streamlit as st
import yaml
from rag import RAGPipeline


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def init_rag():
    config = load_config()
    return RAGPipeline(config), config


def check_prerequisites(config):
    """Check if system is ready."""
    issues = []

    if not config["llm"]["api_key"]:
        issues.append("DeepSeek API Key 未配置。请在 `config.yaml` 的 `llm.api_key` 字段填入密钥。")

    try:
        client = chromadb.PersistentClient(path=config["vectorstore"]["persist_dir"])
        collection = client.get_collection(config["vectorstore"]["collection_name"])
        if collection.count() == 0:
            issues.append("ChromaDB 为空。请先运行 `python ingest.py` 导入教材。")
    except Exception:
        issues.append("ChromaDB 未初始化。请先运行 `python ingest.py` 导入教材。")

    return issues


def render_sidebar(config, rag):
    with st.sidebar:
        st.header("Course QA")
        st.caption(config["course"]["description"])

        st.divider()
        st.markdown(f"**课程**: {config['course']['name']}")
        st.markdown(f"**模型**: {config['llm']['model']}")
        st.markdown(f"**教材 chunks**: {rag.collection.count()}")
        st.markdown(f"**检索 Top-K**: {config['retrieval']['top_k']}")

        st.divider()
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_sources(chunks):
    with st.expander(f"来源 ({len(chunks)})"):
        for i, c in enumerate(chunks):
            meta = c["metadata"]
            score_display = f"精排 {c['rerank_score']:.2f}" if "rerank_score" in c else f"相似度 {1 - c['distance']:.2f}"
            st.markdown(
                f"**[{i+1}]** {meta['chapter']} · 第{meta['page_start']}页 · "
                f"{score_display}"
            )
            st.caption(c["text"][:200] + ("..." if len(c["text"]) > 200 else ""))
            if i < len(chunks) - 1:
                st.divider()


def main():
    st.set_page_config(
        page_title="Course QA",
        page_icon="📚",
        layout="wide",
    )

    config = load_config()
    issues = check_prerequisites(config)

    if issues:
        st.title("📚 Course QA")
        for issue in issues:
            st.warning(issue)
        return

    rag, config = init_rag()
    render_sidebar(config, rag)

    st.title(f"📚 {config['course']['name']} 助教")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "chunks" in msg:
                render_sources(msg["chunks"])

    # Chat input
    if query := st.chat_input("输入关于教材的问题..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate response (Feature 2: streaming with history)
        with st.chat_message("assistant"):
            with st.spinner("检索教材中..."):
                chunks = rag.retrieve(query)

            # Feature 1+2: stream with conversation history (exclude current user msg)
            placeholder = st.empty()
            full_response = ""
            clean_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            for token in rag.generate_stream(query, chunks, history=clean_history):
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            render_sources(chunks)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "chunks": chunks,
        })


if __name__ == "__main__":
    main()
