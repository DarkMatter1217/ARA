import os
import pickle
import re
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
OUTPUT_DIR = "data/faiss_index"
INDEX_PATH = os.path.join(OUTPUT_DIR, "index.faiss")
DOCSTORE_PATH = os.path.join(OUTPUT_DIR, "docstore.pkl")


def normalize_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)


def build_faiss_index(chunks: List[str]):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def save_artifacts(index, chunks: List[str]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(chunks, f)


def run_preprocessing(raw_text: str):
    print("[Preprocessing] Normalizing text...")
    clean_text = normalize_text(raw_text)

    print("[Preprocessing] Chunking text...")
    chunks = chunk_text(clean_text)
    print(f"[Preprocessing] Created {len(chunks)} chunks")

    print("[Preprocessing] Building FAISS index...")
    index = build_faiss_index(chunks)

    print("[Preprocessing] Saving artifacts...")
    save_artifacts(index, chunks)

    print("[Preprocessing] Done.")
