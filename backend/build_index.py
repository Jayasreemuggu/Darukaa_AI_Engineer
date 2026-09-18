from pathlib import Path
import json
import pickle

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KNOWLEDGE_DIR = Path("data/knowledge")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

documents = []
metadata = []

for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        continue

    documents.append(text)
    metadata.append({
        "source_file": file_path.name,
        "title": text.splitlines()[0].replace("TITLE:", "").strip()
    })

if not documents:
    raise RuntimeError("No knowledge documents found.")

print(f"Documents loaded: {len(documents)}")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

embeddings = vectorizer.fit_transform(documents).toarray().astype("float32")

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.maximum(norms, 1e-12)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_DIR / "knowledge.faiss"))

with open(INDEX_DIR / "documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Embedding dimension: {dimension}")
print(f"Vectors indexed: {index.ntotal}")
print("FAISS index: data/index/knowledge.faiss")
print("Documents: data/index/documents.pkl")
print("Metadata: data/index/metadata.json")
print("Vectorizer: data/index/vectorizer.pkl")
print("RAG INDEX READY")