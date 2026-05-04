"""PDF ingestion pipeline: parse, chunk, and store in ChromaDB."""

import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    print(f"Course: {config['course']['name']}")
    print("Ingestion pipeline - not yet implemented")
