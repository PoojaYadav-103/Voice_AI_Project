"""
RAG (Retrieval-Augmented Generation) Module
Uses FAISS + sentence-transformers for local vector search.
No external vector DB needed.
"""

import logging
import os
import pickle
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Lightweight in-process RAG using FAISS.
    
    Usage:
        retriever = RAGRetriever()
        retriever.add_documents([
            {"id": "1", "text": "Paris is the capital of France."},
            {"id": "2", "text": "The Eiffel Tower is in Paris."},
        ])
        results = retriever.retrieve("What is in Paris?", top_k=2)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._documents: List[Dict] = []

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers faiss-cpu"
                )

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the index.
        Each document must have at least a 'text' field.
        Optional fields: 'id', 'source', 'metadata'
        """
        import numpy as np

        self._load_model()

        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")

        texts = [doc["text"] for doc in documents]
        embeddings = self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim (normalized)
        self._index.add(embeddings)
        self._documents = documents

        logger.info(f"RAG index built: {len(documents)} documents, dim={dim}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return top-k most relevant documents for query."""
        if self._index is None or not self._documents:
            return []

        import numpy as np

        self._load_model()
        query_emb = self._model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype=np.float32)

        scores, indices = self._index.search(query_emb, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.3:  # similarity threshold
                doc = dict(self._documents[idx])
                doc["score"] = float(score)
                results.append(doc)

        return results

    def save(self, path: str):
        """Persist index to disk."""
        import faiss
        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.docs.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        logger.info(f"RAG index saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        import faiss
        self._index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.docs.pkl", "rb") as f:
            self._documents = pickle.load(f)
        self._load_model()
        logger.info(f"RAG index loaded: {len(self._documents)} docs")


# ──────────────────────────────────────────────
# Sample knowledge base loader
# ──────────────────────────────────────────────

def load_sample_knowledge_base() -> RAGRetriever:
    """Create a sample RAG retriever with demo knowledge."""
    retriever = RAGRetriever()
    sample_docs = [
        {
            "id": "1",
            "text": "SniperThink is an AI company focused on building intelligent automation tools.",
            "source": "company_info",
        },
        {
            "id": "2",
            "text": "Our flagship product uses machine learning to analyze market trends in real time.",
            "source": "product_info",
        },
        {
            "id": "3",
            "text": "The system supports voice-based interactions with sub-second latency for enterprise clients.",
            "source": "technical_specs",
        },
        {
            "id": "4",
            "text": "To get started, users must sign up on our platform and complete the onboarding workflow.",
            "source": "user_guide",
        },
        {
            "id": "5",
            "text": "Pricing plans start at $49/month for startups and scale based on usage.",
            "source": "pricing",
        },
    ]
    retriever.add_documents(sample_docs)
    return retriever


def load_text_file(filepath: str, chunk_size: int = 500) -> RAGRetriever:
    """
    Load a plain text file and chunk it into a RAG index.
    Useful for loading custom knowledge bases.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple chunking by characters with overlap
    chunks = []
    overlap = 50
    for i in range(0, len(content), chunk_size - overlap):
        chunk_text = content[i : i + chunk_size].strip()
        if chunk_text:
            chunks.append({"id": str(i), "text": chunk_text, "source": filepath})

    retriever = RAGRetriever()
    retriever.add_documents(chunks)
    return retriever
