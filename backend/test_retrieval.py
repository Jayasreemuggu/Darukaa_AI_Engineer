from pathlib import Path
import json
import pickle

import faiss
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = Path("data/index")

index = faiss.read_index(str(INDEX_DIR / "knowledge.faiss"))

with open(INDEX_DIR / "documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open(INDEX_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


def test_index_loaded():
    assert index.ntotal == len(documents)
    assert index.ntotal == len(metadata)
    assert index.d > 0


def test_retrieval():
    query = "soil organic carbon biodiversity"

    query_embedding = vectorizer.transform([query]).toarray().astype("float32")

    norm = np.linalg.norm(query_embedding)

    if norm > 0:
        query_embedding = query_embedding / norm

    scores, indices = index.search(query_embedding, 3)

    assert indices.shape == (1, 3)
    assert scores.shape == (1, 3)

    for idx in indices[0]:
        assert 0 <= idx < len(documents)
