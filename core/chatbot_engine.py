# core/chatbot_engine.py
import os, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import wikipedia

class ChatbotEngine:
    """
    Enhanced ChatbotEngine:
    - FAISS-based retrieval using sentence-transformers
    - Handles both dataset-related and general questions
    - Dynamic Wikipedia fetch if general knowledge is missing
    - Caches embeddings to FAISS index for speed
    """

    def __init__(self,
                 dataset_index_dir: str = "data/faiss_dataset",
                 general_index_dir: str = "data/faiss_general",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.dataset_index_dir = Path(dataset_index_dir)
        self.dataset_index_dir.mkdir(parents=True, exist_ok=True)
        self.general_index_dir = Path(general_index_dir)
        self.general_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentence Transformer model
        self.model = SentenceTransformer(embed_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Dataset FAISS index
        self.dataset_index_file = self.dataset_index_dir / "faiss.index"
        self.dataset_meta_file = self.dataset_index_dir / "metadata.jsonl"
        self.dataset_index = None
        self.dataset_metadata = []
        self._load_index("dataset")
        
        # General (Wikipedia) FAISS index
        self.general_index_file = self.general_index_dir / "faiss.index"
        self.general_meta_file = self.general_index_dir / "metadata.jsonl"
        self.general_index = None
        self.general_metadata = []
        self._load_index("general")

    def _load_index(self, target="dataset"):
        if target == "dataset":
            idx_file, meta_file = self.dataset_index_file, self.dataset_meta_file
            try:
                if idx_file.exists() and meta_file.exists():
                    self.dataset_index = faiss.read_index(str(idx_file))
                    with open(meta_file, "r", encoding="utf8") as f:
                        self.dataset_metadata = [json.loads(line.strip()) for line in f]
            except Exception:
                self.dataset_index = None
                self.dataset_metadata = []
        else:
            idx_file, meta_file = self.general_index_file, self.general_meta_file
            try:
                if idx_file.exists() and meta_file.exists():
                    self.general_index = faiss.read_index(str(idx_file))
                    with open(meta_file, "r", encoding="utf8") as f:
                        self.general_metadata = [json.loads(line.strip()) for line in f]
            except Exception:
                self.general_index = None
                self.general_metadata = []

    def build_index(self, docs: list, target="dataset", chunk_size=512, overlap=50):
        """
        Build FAISS index from docs
        docs: list of tuples (doc_id, text)
        target: "dataset" or "general"
        """
        chunks, metas = [], []
        for doc_id, text in docs:
            words = text.split()
            i = 0; cid = 0
            while i < len(words):
                chunk_words = words[i:i+chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)
                metas.append({"doc_id": doc_id, "chunk_id": cid, "text": chunk_text})
                cid += 1
                i += chunk_size - overlap
        
        if not chunks:
            return 0
        
        embeds = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True).astype("float32")
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeds)
        
        if target == "dataset":
            faiss.write_index(index, str(self.dataset_index_file))
            with open(self.dataset_meta_file, "w", encoding="utf8") as f:
                for m in metas:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            self.dataset_index = index
            self.dataset_metadata = metas
        else:
            faiss.write_index(index, str(self.general_index_file))
            with open(self.general_meta_file, "w", encoding="utf8") as f:
                for m in metas:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            self.general_index = index
            self.general_metadata = metas
        
        return len(metas)

    def query_index(self, text, target="dataset", top_k=5):
        """
        Query FAISS index
        """
        if target == "dataset":
            index = self.dataset_index
            metadata = self.dataset_metadata
        else:
            index = self.general_index
            metadata = self.general_metadata
        
        if index is None or not metadata:
            return {"error": f"{target} index is empty; ingest docs first."}
        
        qv = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = index.search(qv, top_k)
        res = []
        for s, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i < 0 or i >= len(metadata): continue
            m = metadata[i]
            res.append({"score": float(s), "doc_id": m["doc_id"], "chunk_id": m["chunk_id"], "text": m["text"]})
        return {"query": text, "results": res}

    def fetch_wikipedia_content(self, question, top_pages=3):
        """
        Dynamically fetch Wikipedia content for a question
        """
        docs = []
        try:
            results = wikipedia.search(question, results=top_pages)
            for i, title in enumerate(results):
                page = wikipedia.page(title)
                docs.append((f"wiki_{i}", page.content))
        except:
            pass
        return docs

    def answer(self, text: str, top_k=5, target="dataset"):
        """
        Answer question from dataset or general knowledge
        """
        res = self.query_index(text, target=target, top_k=top_k)
        
        # If no results and target=general, try Wikipedia dynamically
        if "error" in res or not res["results"]:
            if target == "general":
                docs = self.fetch_wikipedia_content(text)
                if docs:
                    self.build_index(docs, target="general")
                    res = self.query_index(text, target="general", top_k=top_k)
        
        # Format the answer
        if "error" in res or not res.get("results"):
            return "No relevant knowledge found."
        
        parts = [f"({r['score']:.3f}) {r['text']}" for r in res["results"]]
        return "\n\n".join(parts)
