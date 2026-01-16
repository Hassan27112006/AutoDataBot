# scripts/ingest_wiki.py
import argparse
from core.chatbot_engine import ChatbotEngine
from pathlib import Path

def load_txts(txt_dir):
    p = Path(txt_dir)
    docs = []
    for f in p.glob("*.txt"):
        docs.append((f.stem, f.read_text(encoding="utf8")))
    return docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-dir", required=True, help="Directory containing .txt wiki pages")
    parser.add_argument("--dataset-index-dir", default="data/faiss_index")
    args = parser.parse_args()

    docs = load_txts(args.txt_dir)
    engine = ChatbotEngine(dataset_index_dir=args.dataset_index_dir)
    count = engine.build_index(docs, target="dataset")
    print(f"Indexed {count} chunks.")
