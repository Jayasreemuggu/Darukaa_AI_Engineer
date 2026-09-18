import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")

index = faiss.read_index(str(INDEX_DIR / "knowledge.faiss"))

with open(INDEX_DIR / "documents.pkl", "rb") as f:
    documents = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "My land has low rainfall, poor soil organic carbon and monoculture farming. How can I improve biodiversity?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_embedding, k=3)

print("\nQUERY:")
print(query)

print("\nRETRIEVED KNOWLEDGE:")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    print(f"\n--- RESULT {rank} | similarity={score:.4f} ---")
    print(documents[idx][:1200])

print("\nRETRIEVAL TEST COMPLETE")
