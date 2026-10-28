from fastapi.testclient import TestClient

from src.rag_service.api import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200


def test_returns_citations():
    response = TestClient(app).post(
        "/ask", json={"question": "How do I run the API?", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert "uvicorn" in body["answer"]


def test_rejects_empty_question():
    response = TestClient(app).post("/ask", json={"question": ""})
    assert response.status_code == 422
