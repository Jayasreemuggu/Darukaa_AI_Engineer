import os
import pickle
import uuid
import json
from pathlib import Path
from typing import Optional

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)

INDEX_DIR = Path("data/index")

index = faiss.read_index(str(INDEX_DIR / "knowledge.faiss"))

with open(INDEX_DIR / "documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
    index_metadata = json.load(f)

with open(INDEX_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/knowledge_sources.json", "r", encoding="utf-8") as f:
    source_metadata = json.load(f)


app = FastAPI(
    title="Darukaa.Earth AI Environmental Intelligence",
    description="Evidence-grounded biodiversity intelligence API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary conversational memory.
# Each session_id stores previous user queries and assistant responses.
conversation_memory = {}


class EnvironmentalData(BaseModel):
    soil_ph: Optional[float] = None
    soil_organic_carbon: Optional[float] = None
    soil_moisture: Optional[float] = None
    land_use: Optional[str] = None
    species_richness: Optional[float] = None
    habitat_diversity: Optional[float] = None
    temperature: Optional[float] = None
    rainfall: Optional[str] = None
    pollution: Optional[str] = None
    deforestation: Optional[str] = None
    crop: Optional[str] = None
    region: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5)
    session_id: Optional[str] = None
    environmental_data: Optional[EnvironmentalData] = None
    top_k: int = Field(default=3, ge=1, le=5)


def retrieve(query: str, k: int = 3):
    query_embedding = vectorizer.transform([query]).toarray().astype("float32")

    norm = (query_embedding ** 2).sum() ** 0.5
    if norm > 0:
        query_embedding = query_embedding / norm

    scores, indices = index.search(query_embedding, k)

    evidence = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(documents):
            continue

        meta = index_metadata[idx]
        source_key = Path(meta["source_file"]).stem
        source_info = source_metadata.get(source_key, {})

        evidence.append({
            "similarity": round(float(score), 4),
            "title": source_info.get("title", meta.get("title", "Unknown")),
            "source": source_info.get("organization", "Unknown"),
            "source_url": source_info.get("url", ""),
            "content": documents[idx]
        })

    return evidence


def find_missing_information(data: Optional[EnvironmentalData]):
    if data is None:
        return [
            "soil organic carbon",
            "rainfall pattern",
            "land use or crop type",
            "soil moisture"
        ]

    missing = []

    if data.soil_organic_carbon is None:
        missing.append("soil organic carbon")

    if data.rainfall is None:
        missing.append("rainfall pattern")

    if data.land_use is None and data.crop is None:
        missing.append("land use or crop type")

    if data.soil_moisture is None:
        missing.append("soil moisture")

    return missing


def build_context(data: Optional[EnvironmentalData]):
    if data is None:
        return ""

    values = data.model_dump(exclude_none=True)

    if not values:
        return ""

    return "\nSTRUCTURED ENVIRONMENTAL DATA:\n" + "\n".join(
        f"- {key}: {value}"
        for key, value in values.items()
    )


def build_memory_context(session_id: str):
    history = conversation_memory.get(session_id, [])

    if not history:
        return ""

    recent_history = history[-5:]

    return "\nCONVERSATION HISTORY:\n" + "\n\n".join(
        f"USER: {item['user']}\nASSISTANT: {item['assistant']}"
        for item in recent_history
    )


def build_variable_interactions(data: Optional[EnvironmentalData]):
    if data is None:
        return []

    interactions = []

    soc = data.soil_organic_carbon
    moisture = data.soil_moisture
    rainfall = (data.rainfall or "").lower()
    land_use = (data.land_use or "").lower()
    crop = (data.crop or "").lower()

    # Low rainfall + low soil moisture
    if rainfall == "low" and moisture is not None and moisture < 30:
        interactions.append({
            "variables": ["rainfall", "soil_moisture"],
            "relationship": "Low rainfall combined with low soil moisture indicates a water-availability constraint.",
            "implication": "Interventions should consider water demand and moisture conservation."
        })

    # Low rainfall + agroforestry/tree introduction
    if rainfall == "low" and crop and ("crop" in land_use or land_use == "cropland"):
        interactions.append({
            "variables": ["rainfall", "land_use", "crop"],
            "relationship": "Introducing trees into low-rainfall cropland creates potential tree-crop competition for water, light, and nutrients.",
            "implication": "Tree species selection and spatial arrangement should be adapted to local water availability."
        })

    # Low SOC + low moisture
    if soc is not None and soc < 1.0 and moisture is not None and moisture < 30:
        interactions.append({
            "variables": ["soil_organic_carbon", "soil_moisture"],
            "relationship": "Low soil organic carbon and low soil moisture represent interacting soil-resource constraints.",
            "implication": "Soil-cover and organic-matter management should be considered alongside biodiversity interventions."
        })

    # Monoculture / single crop + biodiversity
    if crop:
        interactions.append({
            "variables": ["crop", "biodiversity"],
            "relationship": "A single-crop system provides less planned crop diversity than a diversified cropping system.",
            "implication": "Crop rotation, mixtures, or other biodiversity practices can be evaluated as complementary interventions."
        })

    # Agroforestry interaction
    if any(word in crop for word in ["wheat", "rice", "maize", "cotton"]) and rainfall == "low":
        interactions.append({
            "variables": ["crop", "rainfall", "land_use"],
            "relationship": f"Growing {crop} in a low-rainfall cropland system makes water availability an important constraint when considering additional vegetation.",
            "implication": "Agroforestry design should account for crop water requirements and tree-crop resource competition."
        })

    return interactions


from pathlib import Path
import time

def reason(
    query: str,
    evidence: list,
    data: Optional[EnvironmentalData],
    session_id: str
):
    evidence_context = "\n\n".join(
        f"[Evidence {i+1} | similarity={item['similarity']}]\n"
        f"{item['content']}"
        for i, item in enumerate(evidence)
    )

    structured_context = build_context(data)
    memory_context = build_memory_context(session_id)
    variable_interactions = build_variable_interactions(data)

    interaction_context = "\n".join(
        f"- Variables: {item['variables']}\n"
        f"  Relationship: {item['relationship']}\n"
        f"  Implication: {item['implication']}"
        for item in variable_interactions
    ) if variable_interactions else (
        "No deterministic variable interactions were identified "
        "from the provided structured data."
    )

    prompt = f"""
You are Darukaa.Earth's environmental intelligence engine.

USER QUERY:
{query}

{structured_context}

{memory_context}

DETERMINISTIC VARIABLE INTERACTIONS:
{interaction_context}

RETRIEVED KNOWLEDGE:
{evidence_context}

Use the retrieved knowledge as the evidence base.

Consider previous conversation context when interpreting the current query.

Return exactly these sections:

ASSESSMENT
Explain the environmental factors and how they interact.

RECOMMENDATION
Give specific practical actions.

WHY IT WORKS
Explain the scientific mechanism using the retrieved evidence.

IMPACTED METRICS
List the environmental metrics affected.

TIME HORIZON
State only time horizons supported by the evidence.
If the evidence does not establish a timeframe, say:
"Timeframe requires site-specific assessment."

EVIDENCE
Connect every major recommendation to retrieved evidence.

LIMITATIONS
State uncertainties and what additional information would improve the recommendation.

Rules:
- Do not invent scientific studies.
- Do not invent citations.
- Do not invent numerical improvement percentages.
- Do not invent unsupported timeframes.
- Distinguish evidence from inference.
- Consider multiple environmental variables together.
"""

    try:
        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return response.output_text

    except Exception as error:
        error_text = str(error).lower()

        if "429" not in error_text and "503" not in error_text:
            raise

        fallback_sections = []

        if data:
            fallback_sections.append(
                "ASSESSMENT\n"
                + structured_context
            )

        if variable_interactions:
            recommendations = "\n".join(
                f"- {item['implication']}"
                for item in variable_interactions
            )

            fallback_sections.append(
                "RECOMMENDATION\n"
                + recommendations
            )

            fallback_sections.append(
                "WHY IT WORKS\n"
                "The recommendations are based on deterministic relationships "
                "between the supplied environmental variables and the "
                "retrieved scientific knowledge."
            )

        metrics = []

        if data:
            if data.soil_organic_carbon is not None:
                metrics.append("Soil organic carbon")

            if data.soil_moisture is not None:
                metrics.append("Soil moisture")

            if data.rainfall is not None:
                metrics.append("Water availability")

            if data.land_use:
                metrics.append("Land use")

            if data.crop:
                metrics.append("Crop/agricultural biodiversity")

        fallback_sections.append(
            "IMPACTED METRICS\n"
            + ", ".join(metrics)
            if metrics
            else
            "IMPACTED METRICS\n"
            "Site-specific environmental metrics require assessment."
        )

        fallback_sections.append(
            "TIME HORIZON\n"
            "Timeframe requires site-specific assessment."
        )

        evidence_lines = []

        for i, item in enumerate(evidence, start=1):
            evidence_lines.append(
                f"- Evidence {i}: {item['title']} "
                f"(similarity {item['similarity']})"
            )

        fallback_sections.append(
            "EVIDENCE\n"
            + (
                "\n".join(evidence_lines)
                if evidence_lines
                else "No retrieved evidence was available."
            )
        )

        fallback_sections.append(
            "LIMITATIONS\n"
            "Generative reasoning is temporarily unavailable. "
            "This response uses the retrieved scientific evidence and "
            "deterministic environmental relationships only. "
            "Additional site-specific measurements would improve the analysis."
        )

        return "\n\n".join(fallback_sections)

@app.get("/")
def root():
    return {
        "name": "Darukaa.Earth AI Environmental Intelligence",
        "version": "3.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "knowledge_documents": len(documents),
        "indexed_vectors": index.ntotal,
        "active_sessions": len(conversation_memory)
    }


@app.post("/analyze")
def analyze(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        missing = find_missing_information(request.environmental_data)

        if missing:
            return {
                "status": "needs_more_information",
                "session_id": session_id,
                "message": "More environmental information is required for a site-specific recommendation.",
                "missing_information": missing,
                "suggested_question": (
                    "Could you provide your "
                    + ", ".join(missing)
                    + "?"
                )
            }

        evidence = retrieve(request.query, request.top_k)

        answer = reason(
            request.query,
            evidence,
            request.environmental_data,
            session_id
        )

        conversation_memory.setdefault(session_id, []).append({
            "user": request.query,
            "assistant": answer
        })

        return {
            "status": "complete",
            "session_id": session_id,
            "query": request.query,
            "answer": answer,
            "conversation_turns": len(conversation_memory[session_id]),
            "reasoning_signals": build_variable_interactions(
                request.environmental_data
            ),
            "retrieved_evidence": evidence
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
