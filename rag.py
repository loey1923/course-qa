"""RAG retrieval and answer generation with DeepSeek API."""

import chromadb
import numpy as np
import yaml
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self, config):
        self.config = config

        # Load embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(
            config["embedding"]["model_path"],
            device=config["embedding"]["device"],
        )

        # Connect to ChromaDB
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=config["vectorstore"]["persist_dir"])
        self.collection = client.get_collection(config["vectorstore"]["collection_name"])
        print(f"  Collection '{self.collection.name}' loaded ({self.collection.count()} chunks)")

        # DeepSeek client
        self.llm = OpenAI(
            api_key=config["llm"]["api_key"],
            base_url=config["llm"]["base_url"],
        )

        # Feature 3: BGE Reranker (CrossEncoder)
        self.reranker = None
        reranker_cfg = config.get("reranker", {})
        if reranker_cfg.get("enabled", False) and reranker_cfg.get("model_path"):
            from sentence_transformers import CrossEncoder
            print("Loading reranker model...")
            self.reranker = CrossEncoder(
                reranker_cfg["model_path"],
                max_length=512,
            )
            print("  Reranker loaded")

    def retrieve(self, query):
        """Retrieve top-K relevant chunks, optionally with MMR diversification.

        Bug 1 fix: when use_mmr=true, fetches top_k*3 candidates then applies
        greedy MMR (weighted by mmr_lambda) to select the final top_k results.
        """
        k = self.config["retrieval"]["top_k"]
        use_mmr = self.config["retrieval"].get("use_mmr", False)
        mmr_lambda = self.config["retrieval"].get("mmr_lambda", 0.5)

        query_embedding = self.embed_model.encode([query])

        # Fetch extra candidates when MMR is enabled
        n_candidates = k * 3 if use_mmr else k
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_candidates,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        if use_mmr and len(docs) > k:
            # MMR: re-encode candidates and compute cosine similarities
            cand_embeddings = self.embed_model.encode(docs)
            q = query_embedding[0]
            q_norm = q / (np.linalg.norm(q) + 1e-10)

            # Similarity of each candidate to the query
            norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-10
            cand_normed = cand_embeddings / norms
            sim_to_query = cand_normed @ q_norm

            # Greedy MMR selection
            selected = []
            remaining = list(range(len(docs)))
            for _ in range(k):
                if not remaining:
                    break
                if not selected:
                    # First pick: most similar to query
                    best_idx = max(remaining, key=lambda i: sim_to_query[i])
                else:
                    best_score = -float("inf")
                    best_idx = remaining[0]
                    for i in remaining:
                        # Max similarity to already-selected items
                        max_sim_sel = max(
                            float(cand_normed[i] @ cand_normed[j]) for j in selected
                        )
                        score = mmr_lambda * sim_to_query[i] - (1 - mmr_lambda) * max_sim_sel
                        if score > best_score:
                            best_score = score
                            best_idx = i
                selected.append(best_idx)
                remaining.remove(best_idx)

            docs = [docs[i] for i in selected]
            metas = [metas[i] for i in selected]
            dists = [dists[i] for i in selected]

        chunks = []
        for i in range(len(docs)):
            chunks.append({
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            })

        # Feature 3: rerank with CrossEncoder when enabled
        if self.reranker and chunks:
            pairs = [(query, c["text"]) for c in chunks]
            scores = self.reranker.predict(pairs)
            for c, score in zip(chunks, scores):
                c["rerank_score"] = float(score)
            chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        return chunks

    def _build_messages(self, query, chunks, history=None):
        """Build the LLM messages list, optionally with conversation history.

        Feature 1: when history is provided, inserts the last 3 turns
        (user/assistant alternating) between system prompt and current query.
        """
        context = "\n\n".join(
            f"[来源: {c['metadata']['chapter']}, 第{c['metadata']['page_start']}页]\n{c['text']}"
            for c in chunks
        )

        prompt = f"""你是一个课程助教。根据以下教材内容回答学生问题。
如果教材中没有相关内容，明确告知"教材中未找到相关说明"。

教材内容：
{context}

学生问题：{query}

要求：
1. 引用具体页码作为出处
2. 回答要准确，不要编造
3. 可以结合教材内容进行推理、举例和解释，帮助学生理解，不必局限于逐字复述教材
4. 数学公式使用 LaTeX 格式：行内公式用 $...$，独立公式用 $$...$$"""

        messages = [
            {"role": "system", "content": "你是一个专业的课程助教，擅长根据教材内容回答问题并进行讲解。回答时使用 Markdown 格式，数学公式用 LaTeX。"},
        ]

        # Feature 1: prepend recent conversation history (last 3 turns = 6 messages)
        if history:
            recent = history[-6:]
            for h in recent:
                if h["role"] in ("user", "assistant"):
                    messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, query, chunks, history=None):
        """Generate answer using DeepSeek with retrieved context.

        Feature 1: accepts optional conversation history for multi-turn chat.
        """
        messages = self._build_messages(query, chunks, history)
        response = self.llm.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=messages,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
        )
        return response.choices[0].message.content

    def generate_stream(self, query, chunks, history=None):
        """Stream answer tokens from DeepSeek.

        Feature 2: yields delta.content strings as they arrive.
        """
        messages = self._build_messages(query, chunks, history)
        stream = self.llm.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=messages,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def ask(self, query):
        """Full RAG pipeline: retrieve + generate."""
        print(f"\nQuestion: {query}")
        print("-" * 40)

        chunks = self.retrieve(query)
        print(f"Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks):
            print(f"  [{i}] dist={c['distance']:.4f} ({c['metadata']['chapter']}, p.{c['metadata']['page_start']})")

        answer = self.generate(query, chunks)
        print(f"\nAnswer:\n{answer}")
        return answer


def main():
    config = load_config()
    rag = RAGPipeline(config)

    print("\nReady! Type your question (or 'quit' to exit):")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            rag.ask(query)


if __name__ == "__main__":
    main()
