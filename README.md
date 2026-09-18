# Darukaa.Earth AI Biodiversity Intelligence

An evidence-grounded AI environmental intelligence system that combines Retrieval-Augmented Generation (RAG), structured environmental data, multi-variable reasoning, conversational memory, and scientific evidence to generate practical biodiversity recommendations.

## Problem Statement

Environmental recommendations depend on interactions between multiple factors such as:

- Soil organic carbon
- Soil moisture
- Rainfall
- Land use
- Crop type
- Biodiversity
- Habitat conditions
- Climate conditions

A generic LLM may generate plausible recommendations without clearly demonstrating the scientific basis behind them.

This project addresses this problem by combining structured environmental information with a retrievable scientific knowledge layer and explicit multi-variable reasoning.

## Key Features

- Retrieval-Augmented Generation (RAG)
- FAISS vector similarity search
- SentenceTransformer embeddings
- Gemini-based reasoning
- Structured environmental inputs
- Deterministic multi-variable reasoning
- Conversational session memory
- Evidence provenance
- Evidence-backed recommendations
- Structured environmental assessment
- Uncertainty and limitation handling
- FastAPI REST API
- Interactive Swagger API documentation

## System Architecture

User Query
    |
    v
FastAPI API
    |
    v
Query Understanding
    |
    +-------------------------+
    |                         |
    v                         v
Structured Environmental   Conversation
Data                        Memory
    |                         |
    +------------+------------+
                 |
                 v
       Multi-Variable Reasoning
                 |
                 v
          FAISS Retrieval
                 |
                 v
        Retrieved Evidence
                 |
                 v
         Gemini Reasoning
                 |
                 v
    Evidence-backed Recommendation
                 |
                 v
        Structured API Response

## RAG Pipeline

Scientific Knowledge
        |
        v
Knowledge Documents
        |
        v
SentenceTransformer Embeddings
        |
        v
FAISS Vector Index
        |
        v
User Query
        |
        v
Query Embedding
        |
        v
Similarity Search
        |
        v
Top-K Retrieved Evidence
        |
        v
Gemini Reasoning
        |
        v
Final Recommendation

## Knowledge Base

The current knowledge base contains source-backed summaries from FAO and IPCC material.

### FAO - Soil Organic Cover and Conservation Agriculture

The document covers:

- Soil cover
- Cover crops
- Crop residues
- Soil structure
- Organic matter
- Water infiltration
- Evaporation
- Crop diversification
- Soil biodiversity

Source: https://www.fao.org/conservation-agriculture/in-practice/soil-organic-cover/en/

### FAO - Agricultural Biodiversity

The document covers:

- Crop rotations
- Crop mixtures
- Permanent soil cover
- Agroforestry
- Soil organisms
- Nutrient cycling
- Pollination
- Pest regulation
- Soil moisture
- Ecosystem functions

Source: https://www.fao.org/agriculture/crops/thematic-sitemap/theme/compendium/tools-guidelines/what-is-agricultural-biodiversity/en/

### FAO - Agroforestry and Biodiversity

The document covers:

- Trees with crops and livestock
- Biodiversity
- Ecosystem services
- Soil fertility
- Water regulation
- Tree-crop interactions
- Resource competition
- Landscape resilience

Source: https://www.fao.org/americas/priorities/agricultura-sostenible/agrofesteria/en

### IPCC - Agriculture, Forestry and Other Land Use

Based on IPCC AR6 WGIII Chapter 7.

The document covers:

- Land management
- Climate mitigation
- Biodiversity
- Habitat conservation
- Carbon
- Food production
- Co-benefits
- Trade-offs
- Site-specific implementation

Source: https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-7/

## Structured Environmental Data

The API supports the following environmental variables:

| Variable | Description |
|---|---|
| soil_ph | Soil pH |
| soil_organic_carbon | Soil organic carbon |
| soil_moisture | Soil moisture |
| land_use | Land-use category |
| species_richness | Species richness |
| habitat_diversity | Habitat diversity |
| temperature | Temperature |
| rainfall | Rainfall condition |
| pollution | Pollution condition |
| deforestation | Deforestation condition |
| crop | Crop type |
| region | Geographic/environmental region |

## Multi-Variable Reasoning

The system does not treat environmental variables independently.

A deterministic reasoning layer identifies relationships between environmental variables before the LLM generates the final recommendation.

Examples include:

### Rainfall + Soil Moisture

Low rainfall combined with low soil moisture indicates a water-availability constraint.

### Soil Organic Carbon + Soil Moisture

Low soil organic carbon and low soil moisture represent interacting soil-resource constraints.

### Crop + Biodiversity

A single-crop system provides less planned crop diversity than a diversified cropping system.

### Crop + Rainfall + Land Use

Growing a crop in a low-rainfall cropland system makes water availability an important constraint when considering additional vegetation.

These relationships are passed to the reasoning layer as structured reasoning signals.

The reasoning signals are not treated as scientific evidence by themselves. Scientific claims are supported through retrieved knowledge.

## Conversational Memory

The system supports multi-turn conversations using a session_id.

Example:

User:
My farm has low rainfall and low soil organic carbon.

Assistant:
Provides recommendations based on the provided conditions.

User:
What about adding trees?

Assistant:
Uses the previous conversation context and environmental conditions when interpreting the follow-up question.

Recent conversation history is included in the reasoning context for the same session.

## Evidence Provenance

Each retrieved knowledge item contains:

- Similarity score
- Source title
- Organization
- Source URL
- Retrieved content

Example:

{
    "similarity": 0.4984,
    "title": "Soil Organic Cover and Conservation Agriculture",
    "source": "FAO",
    "source_url": "https://www.fao.org/conservation-agriculture/in-practice/soil-organic-cover/en/"
}

This allows recommendations to be traced back to the retrieved knowledge sources.

## Recommendation Structure

The generated response follows a structured format.

### ASSESSMENT

Explains the environmental conditions and how they interact.

### RECOMMENDATION

Provides specific practical actions.

### WHY IT WORKS

Explains the scientific mechanisms using retrieved evidence.

### IMPACTED METRICS

Identifies environmental metrics potentially affected by the recommendation.

### TIME HORIZON

Provides only evidence-supported time horizons.

If the available evidence does not establish a timeframe, the system reports:

"Timeframe requires site-specific assessment."

### EVIDENCE

Connects major recommendations to retrieved evidence.

### LIMITATIONS

Describes uncertainty, site-specific constraints, and additional information required.

## Example Request

{
    "query": "My farm has low rainfall, low soil organic carbon and wheat monoculture. What should I do?",
    "environmental_data": {
        "soil_organic_carbon": 0.8,
        "soil_moisture": 22,
        "land_use": "cropland",
        "rainfall": "low",
        "crop": "wheat",
        "region": "semi-arid"
    },
    "top_k": 3
}

## Example Processing Flow

Input Environmental Conditions
        |
        v
Variable Interaction Detection
        |
        v
FAISS Knowledge Retrieval
        |
        v
Relevant FAO/IPCC Evidence
        |
        v
Gemini Reasoning
        |
        v
Environmental Assessment
        |
        v
Recommendations
        |
        v
Impacted Metrics
        |
        v
Evidence and Limitations

## API Endpoints

### Root

GET /

Returns basic application information.

### Health Check

GET /health

Returns:

- API status
- Number of knowledge documents
- Number of indexed vectors
- Active sessions

### Environmental Analysis

POST /analyze

Accepts:

- User query
- Environmental data
- Optional session ID
- Number of retrieved documents

Returns:

- Status
- Session ID
- Generated answer
- Conversation turn count
- Reasoning signals
- Retrieved evidence

## Interactive API Documentation

When the backend is running locally:

http://127.0.0.1:8001/docs

FastAPI provides an interactive Swagger interface for testing the API.

## Project Structure

Darukaa_AI_Engineer/
|
+-- backend/
|   +-- main.py
|   +-- build_index.py
|
+-- data/
|   +-- knowledge/
|   |   +-- fao_soil_cover.txt
|   |   +-- fao_agroforestry.txt
|   |   +-- fao_agricultural_biodiversity.txt
|   |   +-- ipcc_afolu.txt
|   |
|   +-- index/
|   |   +-- knowledge.faiss
|   |   +-- documents.pkl
|   |   +-- metadata.json
|   |
|   +-- environment_schema.json
|   +-- knowledge_sources.json
|
+-- tests/
|
+-- .github/
|   +-- workflows/
|
+-- .env
+-- .gitignore
+-- README.md
+-- requirements.txt

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| LLM | Gemini |
| Embeddings | SentenceTransformers |
| Vector Search | FAISS |
| Data Processing | Pandas / NumPy |
| Validation | Pydantic |
| Server | Uvicorn |
| Knowledge Sources | FAO / IPCC |

## Installation

### 1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Darukaa_AI_Engineer

### 2. Create a Virtual Environment

python -m venv .venv

### 3. Activate the Environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

### 4. Install Dependencies

pip install -r requirements.txt

## Environment Configuration

Create a .env file in the project root.

GEMINI_API_KEY=your_gemini_api_key

Do not commit the .env file to GitHub.

The .gitignore file should include:

.env
.venv/
__pycache__/
*.pyc

## Build the Knowledge Index

Run:

python backend/build_index.py

The indexing pipeline:

1. Loads knowledge documents.
2. Generates embeddings.
3. Normalizes embeddings.
4. Creates the FAISS index.
5. Stores document metadata.

Generated files:

data/index/knowledge.faiss
data/index/documents.pkl
data/index/metadata.json

## Run the Backend

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001

Then open:

http://127.0.0.1:8001/docs

## Scientific Grounding

The current knowledge layer uses source-backed summaries based on FAO and IPCC material.

The LLM is not treated as the primary scientific knowledge source.

Instead, the system follows:

Scientific Sources
        |
        v
Knowledge Documents
        |
        v
Embeddings
        |
        v
FAISS Retrieval
        |
        v
Relevant Evidence
        |
        v
Gemini Reasoning
        |
        v
Recommendation

This separates knowledge retrieval from language generation.

## Evidence vs Reasoning

The system separates three concepts.

### Retrieved Evidence

Information retrieved from the scientific knowledge base.

### Reasoning Signals

Structured relationships identified from the supplied environmental variables.

### Generated Recommendation

The final recommendation generated using the retrieved evidence, structured data, reasoning signals, and conversational context.

This separation helps distinguish source-supported information from model reasoning.

## Design Principles

### Evidence Before Generation

Relevant knowledge is retrieved before recommendations are generated.

### Multi-Variable Analysis

Environmental conditions are considered together rather than independently.

### Traceability

Retrieved evidence exposes source metadata and source URLs.

### Explicit Uncertainty

Unsupported numerical claims and unsupported timeframes are avoided.

### Site-Specific Reasoning

Environmental interventions may involve trade-offs, so local environmental conditions are considered.

### Modular Architecture

Retrieval, reasoning, memory, API, and knowledge storage are separated so individual components can be extended independently.

## Current Limitations

- The current knowledge base contains a limited number of source documents.
- Conversation memory is currently stored in process memory.
- The current system does not yet use live geospatial datasets.
- Quantitative prediction of environmental outcomes requires additional validated datasets.
- Specific crop and intervention suitability remains site-specific.
- The current knowledge documents are source-backed summaries rather than complete copies of the original publications.

## Future Improvements

### Knowledge Layer

- Expand the scientific knowledge base.
- Add more environmental reports and research.
- Improve document chunking.
- Add metadata-aware retrieval.
- Implement hybrid keyword and vector retrieval.

### Environmental Intelligence

- Integrate satellite-derived indicators.
- Integrate soil datasets.
- Integrate biodiversity datasets.
- Integrate climate datasets.
- Add geospatial analysis.

### AI

- Confidence estimation.
- Improved evidence-to-recommendation mapping.
- Automated factuality evaluation.
- Advanced multi-objective reasoning.

### Memory

- Persistent database-backed conversation memory.
- User-specific environmental profiles.

### Deployment

- Docker.
- Cloud deployment.
- CI/CD.
- Automated testing.
- Production monitoring.

## Evaluation Alignment

| Challenge Requirement | Project Implementation |
|---|---|
| Reasoning depth | Deterministic multi-variable reasoning + Gemini |
| Scientific grounding | FAO/IPCC knowledge retrieval |
| Knowledge design | Structured knowledge documents + FAISS |
| Conversational intelligence | Session-based conversation memory |
| Evidence-backed recommendations | Retrieved evidence + provenance |
| Multi-metric reasoning | Explicit environmental variable interactions |
| Output clarity | Structured response sections |
| Uncertainty handling | Limitations and unsupported timeframe handling |

## Current Status

The current prototype demonstrates:

- FastAPI backend
- Gemini integration
- SentenceTransformer embeddings
- FAISS vector retrieval
- Source metadata
- Evidence provenance
- Structured environmental inputs
- Multi-variable reasoning signals
- Conversational memory
- Evidence-backed recommendations
- Structured response format
- Swagger API documentation

## Project Goal

The objective is not simply to generate environmental advice using an LLM.

The objective is to build an evidence-grounded environmental intelligence system in which:

Environmental Context
        +
Scientific Knowledge
        +
Multi-Variable Reasoning
        +
Conversational Context
        |
        v
Evidence-backed Environmental Recommendation

The architecture is designed to make AI-generated biodiversity recommendations more traceable, explainable, and scientifically grounded.