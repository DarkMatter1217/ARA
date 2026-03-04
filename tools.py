from typing import List
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/index.faiss")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "data/faiss_index/docstore.pkl")

_embedding_model = None
_faiss_index = None
_docstore = None


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


def vector_search(query: str, top_k: int = 5) -> List[str]:
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


def sample_size_check(n: int, threshold: int = 30) -> bool:
    return n is not None and n >= threshold


def bias_scan(text: str) -> List[str]:
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
# -------------------------------
# ARA Evidence FAISS (separate)
# -------------------------------
ARA_EVIDENCE_DIR = "data/ara_evidence_faiss"
ARA_EVIDENCE_INDEX_PATH = os.path.join(ARA_EVIDENCE_DIR, "index.faiss")
ARA_EVIDENCE_DOCSTORE_PATH = os.path.join(ARA_EVIDENCE_DIR, "docstore.pkl")

def classify_quality(url: str):
    u = url.lower()
    if "arxiv.org" in u or "nature.com" in u or "sciencedirect.com" in u:
        return "HIGH"
    if ".edu" in u or ".gov" in u or "github.com" in u:
        return "MEDIUM"
    return "LOW"


from exa_py import Exa

EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise RuntimeError("EXA_API_KEY not set")

_exa = Exa(EXA_API_KEY)

def exa_search(query: str, max_results: int = 5):
    response = _exa.search(
        query,
        num_results=max_results,
    )

    results = []

    for r in response.results:
        text = r.text if hasattr(r, "text") and r.text else ""

        results.append({
            "url": r.url,
            "title": r.title,
            "content": text,
            "quality": classify_quality(r.url)
        })

    return results

def init_or_load_evidence_store():
    os.makedirs(ARA_EVIDENCE_DIR, exist_ok=True)

    model = _load_embedding_model()

    if os.path.exists(ARA_EVIDENCE_INDEX_PATH):
        index = faiss.read_index(ARA_EVIDENCE_INDEX_PATH)
        with open(ARA_EVIDENCE_DOCSTORE_PATH, "rb") as f:
            docstore = pickle.load(f)
    else:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        docstore = []

    return index, docstore


def save_evidence_store(index, docstore):
    faiss.write_index(index, ARA_EVIDENCE_INDEX_PATH)
    with open(ARA_EVIDENCE_DOCSTORE_PATH, "wb") as f:
        pickle.dump(docstore, f)


def add_sources_to_evidence_store(sources):
    index, docstore = init_or_load_evidence_store()
    model = _load_embedding_model()

    for src in sources:
        from preprocessing import chunk_text
        chunks =chunk_text(src["content"])
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            embedding = model.encode([chunk], normalize_embeddings=True)
            index.add(embedding)
            docstore.append({
                "content": chunk,
                "url": src["url"],
                "quality": src["quality"]
            })

    save_evidence_store(index, docstore)


def retrieve_evidence_context(query: str, top_k: int = 5):
    index, docstore = init_or_load_evidence_store()
    model = _load_embedding_model()

    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(docstore[idx])

    return results



# ===============================
# DOCUMENT FAISS STORE (Baseline)
# ===============================

DOCUMENT_FAISS_DIR = "data/document_faiss"

_document_vector_store = None

def load_document_vector_store():
    global _document_vector_store

    if _document_vector_store is not None:
        return _document_vector_store

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    if os.path.exists(DOCUMENT_FAISS_DIR):
        _document_vector_store = FAISS.load_local(
            DOCUMENT_FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        raise ValueError("Document FAISS store not found. Build it first.")

    return _document_vector_store


def document_vector_search(query: str, top_k: int = 5):
    store = load_document_vector_store()
    docs = store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

def reset_evidence_store():
    import shutil
    if os.path.exists(ARA_EVIDENCE_DIR):
        shutil.rmtree(ARA_EVIDENCE_DIR)