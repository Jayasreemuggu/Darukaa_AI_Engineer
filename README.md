# Darukaa.Earth — AI Biodiversity & Environmental Intelligence

An evidence-grounded AI system for environmental and biodiversity intelligence. The system combines structured environmental inputs, deterministic multi-variable reasoning, Retrieval-Augmented Generation (RAG), scientific evidence, conversational context, and Gemini-based reasoning to produce actionable environmental recommendations.

## Live Demo

- Frontend: https://jayasreemuggu.github.io/Darukaa_AI_Engineer/frontend/
- Backend API: https://darukaa-ai-engineer.onrender.com
- API Documentation: https://darukaa-ai-engineer.onrender.com/docs
- GitHub Repository: https://github.com/Jayasreemuggu/Darukaa_AI_Engineer

## Problem

Environmental decisions often require reasoning across multiple interacting variables such as:

- Soil organic carbon
- Soil moisture
- Rainfall
- Land use
- Crop systems
- Biodiversity
- Habitat and ecological conditions

A generic language model can produce plausible advice, but it does not guarantee that recommendations are grounded in environmental evidence.

Darukaa.Earth addresses this by combining structured environmental data, retrieval, deterministic environmental reasoning, and evidence-grounded AI generation.

## Key Features

### Evidence-Grounded RAG

The system retrieves relevant information from a curated environmental knowledge base before generating recommendations.

Current knowledge sources include:

- FAO — Soil Organic Cover and Conservation Agriculture
- FAO — Agroforestry and Biodiversity
- FAO — Agricultural Biodiversity
- IPCC AR6 WGIII — AFOLU

Retrieved evidence is displayed to the user with source information.

### Multi-Variable Environmental Reasoning

The system reasons across interacting environmental variables, including:

- Soil organic carbon ↔ soil moisture
- Soil moisture ↔ water availability
- Land use ↔ biodiversity
- Crop diversity ↔ agricultural biodiversity
- Agroforestry ↔ habitat and soil conditions

### Structured Environmental Inputs

The API accepts structured environmental information such as:

- Soil organic carbon
- Soil moisture
- Rainfall
- Land use
- Crop
- Region

Example:

{
  "soil_organic_carbon": 0.8,
  "soil_moisture": 22,
  "land_use": "cropland",
  "rainfall": "low",
  "crop": "wheat",
  "region": "semi-arid"
}

### Conversational Intelligence

The system can:

- Identify missing environmental information
- Ask for clarification when important variables are absent
- Maintain recent conversational context
- Adapt recommendations to supplied environmental conditions

### Evidence-Backed Recommendations

Recommendations contain:

- Recommended action
- Scientific reasoning
- Impacted environmental metrics
- Time horizon
- Supporting evidence
- Limitations

### Explainable Reasoning Signals

The API exposes structured reasoning signals showing how environmental variables contribute to the final recommendation.

## System Architecture

User
  |
  v
Web Frontend
  |
  v
FastAPI Backend
  |
  +--> Query Understanding
  |
  +--> Structured Environmental Data
  |
  +--> Missing Information Detection
  |
  +--> Environmental Interaction Reasoning
  |
  +--> TF-IDF Retrieval
  |       |
  |       v
  |    FAISS Index
  |       |
  |       v
  |    Scientific Knowledge
  |
  +--> Conversational Memory
  |
  v
Gemini Reasoning Layer
  |
  v
Evidence-Grounded Recommendation
  |
  +--> Recommendation
  +--> Impacted Metrics
  +--> Time Horizon
  +--> Evidence
  +--> Reasoning Signals
  +--> Limitations

## RAG Pipeline

Scientific Documents
        |
        v
Text Processing
        |
        v
TF-IDF Vectorization
        |
        v
Normalized Vectors
        |
        v
FAISS Similarity Index
        |
        v
Relevant Evidence Retrieval
        |
        v
Gemini Reasoning
        |
        v
Grounded Environmental Recommendation

The retrieval layer is implemented separately from the language model so that the model is not the sole source of environmental knowledge.

## Retrieval Architecture

The deployed version uses a lightweight retrieval architecture:

- TF-IDF for lexical feature extraction
- FAISS for similarity search
- Normalized vectors with inner-product similarity
- Local indexed scientific documents
- Metadata stored alongside the indexed documents

This design avoids loading large transformer models during deployment and keeps the application suitable for a constrained cloud environment.

## Scientific Grounding

The knowledge base contains information from recognized environmental and scientific organizations, including FAO and IPCC.

Each retrieved document retains source metadata so recommendations can be traced back to supporting evidence.

## API

### Health Check

GET /health

### Environmental Analysis

POST /analyze

Example request:

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

The response contains:

- AI recommendation
- Environmental assessment
- Reasoning signals
- Retrieved evidence
- Impacted metrics
- Time horizon
- Limitations

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- FAISS
- Google Gemini API

### AI / Retrieval

- Retrieval-Augmented Generation
- TF-IDF
- FAISS similarity search
- Structured environmental reasoning
- Conversational memory
- Evidence provenance

### Frontend

- HTML
- CSS
- JavaScript
- REST API integration

### Deployment

- GitHub
- GitHub Actions
- Render
- GitHub Pages

## Project Structure

Darukaa_AI_Engineer/
|
+-- backend/
|   +-- main.py
|   +-- build_index.py
|
+-- data/
|   +-- knowledge/
|   +-- index/
|
+-- frontend/
|   +-- index.html
|
+-- tests/
|   +-- test_api.py
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml
|
+-- requirements.txt
+-- runtime.txt
+-- .gitignore
+-- README.md

## Local Setup

Clone the repository:

git clone https://github.com/Jayasreemuggu/Darukaa_AI_Engineer.git

cd Darukaa_AI_Engineer

Create the virtual environment:

python -m venv .venv

Install dependencies:

pip install -r requirements.txt

Create a .env file containing:

GEMINI_API_KEY=your_api_key

Build the knowledge index:

python backend/build_index.py

Run the backend:

python -m uvicorn backend.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

## Frontend Local Setup

From the project directory:

cd frontend

python -m http.server 5500

Open:

http://localhost:5500

The frontend communicates with the deployed FastAPI backend.

## Testing

Run:

pytest tests/test_api.py -v

The backend test suite covers the core API behavior and uses a mocked reasoning layer during tests.

## Deployment

The project uses a separated frontend/backend deployment architecture.

Frontend:
GitHub Pages

Backend:
Render

The frontend sends requests to the deployed FastAPI API.

## CORS

The backend includes CORS configuration so the GitHub Pages frontend can communicate with the deployed FastAPI service.

## Evaluation Alignment

### Reasoning — 30%

The system combines deterministic environmental interaction logic with Gemini reasoning rather than relying only on free-form generation.

### Scientific Grounding — 25%

Recommendations are connected to retrieved evidence from FAO and IPCC sources.

### Knowledge Design — 20%

The knowledge layer is stored separately from the model and indexed for retrieval.

### Conversational Intelligence — 15%

The backend supports context-aware analysis and missing-information detection.

### Output Clarity — 10%

The response separates:

- Recommendation
- Impacted metrics
- Time horizon
- Evidence
- Reasoning signals
- Limitations

## Example Environmental Scenario

Input:

"My farm has low rainfall, low soil organic carbon and wheat monoculture. What should I do?"

Environmental conditions:

- Soil organic carbon: 0.8
- Soil moisture: 22
- Rainfall: Low
- Land use: Cropland
- Crop: Wheat
- Region: Semi-arid

The system identifies interacting constraints involving water availability, soil condition and crop diversity, retrieves relevant scientific evidence, and generates recommendations such as soil-cover/residue management, crop diversification and targeted agroforestry where appropriate.

## Limitations

- The current knowledge base is intentionally small and curated for the prototype.
- TF-IDF retrieval is lexical rather than transformer-based semantic retrieval.
- Environmental recommendations are decision-support outputs and should be interpreted alongside site-specific expertise.
- More geographic and temporal data would improve location-specific recommendations.
- Larger validated environmental datasets could support stronger quantitative prediction models.

## Future Improvements

- Expand the environmental and biodiversity knowledge base
- Add larger environmental datasets
- Add geospatial coordinate support
- Integrate satellite-derived environmental indicators
- Add stronger semantic retrieval
- Add evidence reranking
- Add PostgreSQL/PostGIS
- Add retrieval-quality evaluation
- Add uncertainty estimation
- Add environmental time-series analysis
- Add automated monitoring and observability

## Security

- API keys are stored through environment variables.
- The .env file is excluded from Git.
- Sensitive credentials are not committed to the repository.

## Repository

https://github.com/Jayasreemuggu/Darukaa_AI_Engineer

## Live Services

Frontend:
https://jayasreemuggu.github.io/Darukaa_AI_Engineer/frontend/

Backend:
https://darukaa-ai-engineer.onrender.com

Swagger:
https://darukaa-ai-engineer.onrender.com/docs

## Author

Developed as an AI Engineer internship challenge project for Darukaa.Earth.
