# scripts/build_wiki_dataset.py
import os
import wikipedia
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# STEP 1 — SETTINGS
# ------------------------------------------------------------

SAVE_DIR = "data/wiki_pages"
INDEX_DIR = "data/faiss_index"
CHUNK_SIZE = 500  # characters per chunk
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Topics to download (ADD MORE ANY TIME)
TOPICS = [
    "Python (programming language)",
    "Machine learning",
    "Artificial intelligence",
    "Data science",
    "Neural network",
    "Flask (web framework)"
]

# ------------------------------------------------------------
# STEP 2 — DOWNLOAD WIKIPEDIA PAGES
# ------------------------------------------------------------

def fetch_wiki_pages():
    print("\n=== Fetching Wikipedia Pages ===\n")
    for topic in TOPICS:
        try:
            page = wikipedia.page(topic)
            filename = topic.replace("/", "_").replace(" ", "_") + ".txt"
            path = os.path.join(SAVE_DIR, filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(page.content)

            print(f"[✓] Saved: {filename}")

        except Exception as e:
            print(f"[x] Error fetching {topic}: {e}")


# ------------------------------------------------------------
# STEP 3 — CHUNK TEXT FILES
# ------------------------------------------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ------------------------------------------------------------
# STEP 4 — BUILD FAISS INDEX
# ------------------------------------------------------------

def build_faiss_index():
    print("\n=== Building FAISS Index ===\n")

    # Load embedder
    model = SentenceTransformer(EMBED_MODEL)

    documents = []
    metadata = []

    # Load all .txt files
    for fname in os.listdir(SAVE_DIR):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(SAVE_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        documents.extend(chunks)

        metadata.extend([{"source": fname}] * len(chunks))

    print(f"Total chunks: {len(documents)}")

    # Compute embeddings
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS L2 index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(INDEX_DIR, "wiki_index.faiss"))

    # Save metadata
    np.save(os.path.join(INDEX_DIR, "wiki_docs.npy"), documents, allow_pickle=True)
    np.save(os.path.join(INDEX_DIR, "wiki_meta.npy"), metadata, allow_pickle=True)

    print("\n[✓] FAISS index built and saved!")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    fetch_wiki_pages()
    build_faiss_index()
    print("\n=== DONE ===")
