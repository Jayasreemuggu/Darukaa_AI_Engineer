from pathlib import Path
import json
import pickle

import faiss
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_DIR / "knowledge.faiss"))

with open(INDEX_DIR / "documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Embedding dimension: {dimension}")
print(f"Vectors indexed: {index.ntotal}")
print("FAISS index: data/index/knowledge.faiss")
print("Documents: data/index/documents.pkl")
print("Metadata: data/index/metadata.json")
print("RAG INDEX READY")
