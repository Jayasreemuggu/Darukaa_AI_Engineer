import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add your Gemini API key to the .env file."
    )

client = genai.Client(api_key=API_KEY)

INDEX_DIR = Path("data/index")

index = faiss.read_index(str(INDEX_DIR / "knowledge.faiss"))

with open(INDEX_DIR / "documents.pkl", "rb") as f:
    documents = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, k=3):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    return [
        {
            "score": float(score),
            "content": documents[idx]
        }
        for score, idx in zip(scores[0], indices[0])
    ]


def generate_recommendation(query):
    results = retrieve(query)

    evidence = "\n\n".join(
        f"[Evidence {i+1} | similarity={r['score']:.4f}]\n{r['content']}"
        for i, r in enumerate(results)
    )

    prompt = f"""
You are an environmental intelligence system.

Reason ONLY from the retrieved evidence below.
Do not invent scientific studies, statistics, percentages, or citations.

USER QUERY:
{query}

RETRIEVED EVIDENCE:
{evidence}

Produce exactly these sections:

1. ASSESSMENT
Identify the environmental factors and explain how they interact.

2. RECOMMENDATION
Give specific practical actions.

3. WHY IT WORKS
Explain the scientific mechanism using the retrieved evidence.

4. IMPACTED METRICS
List the environmental metrics expected to change.

5. TIME HORIZON
Classify effects as short-term, medium-term, or long-term.

6. EVIDENCE
Identify which retrieved evidence supports each major recommendation.

7. LIMITATIONS
State what additional information is needed for a site-specific recommendation.

Be conservative. If the retrieved evidence does not establish a quantitative value,
do not create one.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text


if __name__ == "__main__":
    query = (
        "My land has low rainfall, poor soil organic carbon and "
        "monoculture farming. How can I improve biodiversity?"
    )

    print("\n" + "=" * 70)
    print("DARUKAA.EARTH ENVIRONMENTAL INTELLIGENCE")
    print("=" * 70)

    print("\nQUERY:")
    print(query)

    print("\nGENERATED ANALYSIS:")
    print(generate_recommendation(query))

    print("\n" + "=" * 70)
    print("REASONING TEST COMPLETE")
    print("=" * 70)
