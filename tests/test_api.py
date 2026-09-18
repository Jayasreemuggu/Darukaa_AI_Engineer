from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_missing_environmental_data():
    response = client.post(
        "/analyze",
        json={"query": "How can I improve biodiversity on my farm?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "needs_more_information"
    assert "missing_information" in data


@patch(
    "backend.main.reason",
    return_value=(
        "ASSESSMENT: The supplied conditions indicate water and soil-resource constraints.\n"
        "RECOMMENDATION: Improve soil cover and diversify the cropping system.\n"
        "WHY IT WORKS: These practices can support soil function and biodiversity.\n"
        "IMPACTED METRICS: Soil organic carbon, soil moisture, biodiversity.\n"
        "TIME HORIZON: Depends on site conditions and management.\n"
        "EVIDENCE: Grounded in the retrieved knowledge sources.\n"
        "LIMITATIONS: Site-specific monitoring is required."
    ),
)
def test_analyze_reasoning_signals(mock_reason):
    response = client.post(
        "/analyze",
        json={
            "query": "My farm has low rainfall and low soil moisture.",
            "environmental_data": {
                "soil_organic_carbon": 0.8,
                "soil_moisture": 22,
                "land_use": "cropland",
                "rainfall": "low",
                "crop": "wheat",
                "region": "semi-arid",
            },
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "complete"
    assert "reasoning_signals" in data
    assert len(data["reasoning_signals"]) > 0
    assert "retrieved_evidence" in data
    assert len(data["retrieved_evidence"]) > 0

    mock_reason.assert_called_once()