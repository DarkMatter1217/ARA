# tools.py

from typing import List
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


# -------------------------
# Global, Read-Only Objects
# -------------------------

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/index.faiss")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "data/faiss_index/docstore.pkl")

_embedding_model = None
_faiss_index = None
_docstore = None


# -------------------------
# Lazy Loaders
# -------------------------

def _load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
)

    return _embedding_model


def _load_faiss():
    global _faiss_index, _docstore
    if _faiss_index is None:
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCSTORE_PATH, "rb") as f:
            _docstore = pickle.load(f)
    return _faiss_index, _docstore


# -------------------------
# Vector Search Tool
# -------------------------

def vector_search(query: str, top_k: int = 5) -> List[str]:
    """
    Performs similarity search over FAISS index.
    Returns top_k relevant text chunks.
    """
    model = _load_embedding_model()
    index, docstore = _load_faiss()

    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(docstore[idx])

    return results


# -------------------------
# Simple Rule-Based Tools
# -------------------------

def sample_size_check(n: int, threshold: int = 30) -> bool:
    """
    Returns True if sample size is acceptable.
    """
    return n is not None and n >= threshold


def bias_scan(text: str) -> List[str]:
    """
    Scans text for conflict-of-interest keywords.
    """
    keywords = [
        "conflict of interest",
        "funded by",
        "sponsored by",
        "financial support",
        "industry funded",
    ]

    found = []
    lowered = text.lower()
    for kw in keywords:
        if kw in lowered:
            found.append(kw)

    return found
