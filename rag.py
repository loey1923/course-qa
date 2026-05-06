"""RAG retrieval and answer generation with DeepSeek API."""

import chromadb
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

    def retrieve(self, query):
        """Retrieve top-K relevant chunks."""
        k = self.config["retrieval"]["top_k"]
        query_embedding = self.embed_model.encode([query])

        kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": k,
        }

        results = self.collection.query(**kwargs)

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def generate(self, query, chunks):
        """Generate answer using DeepSeek with retrieved context."""
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

        response = self.llm.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=[
                {"role": "system", "content": "你是一个专业的课程助教，擅长根据教材内容回答问题并进行讲解。回答时使用 Markdown 格式，数学公式用 LaTeX。"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
        )
        return response.choices[0].message.content

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
