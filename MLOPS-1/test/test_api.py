import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import app

# Create a reusable test client
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.6,
        "petal_width": 0.5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert result["prediction"] in [0, 1, 2]


def test_invalid_iris_input():
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.6,
        "petal_width": 0.5
    }

    response = client.post("/predict", json=payload)

    # Expects FastAPI/Pydantic to reject negative measurements
    assert response.status_code == 422