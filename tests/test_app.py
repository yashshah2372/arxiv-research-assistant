"""Offline tests: no API key or vector DB needed."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from app import app, get_answerer  # noqa: E402


def fake_answer(question, k=3):
    metas = [{"title": f"Paper {i}", "url": f"https://arxiv.org/abs/{i}"} for i in range(1, k + 1)]
    return f"Answer to: {question} [1]", metas


def client_with(answerer):
    app.dependency_overrides[get_answerer] = lambda: answerer
    return TestClient(app)


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_numbered_sources():
    r = client_with(fake_answer).post("/ask", json={"question": "How is RAG evaluated?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Answer to: How is RAG evaluated?")
    assert [s["number"] for s in body["sources"]] == [1, 2]
    assert body["sources"][0]["url"] == "https://arxiv.org/abs/1"


def test_ask_validates_input():
    c = client_with(fake_answer)
    assert c.post("/ask", json={"question": "hi"}).status_code == 422
    assert c.post("/ask", json={"question": "valid question", "k": 50}).status_code == 422


def test_upstream_failure_becomes_502():
    def broken(question, k=3):
        raise RuntimeError("LLM down")

    r = client_with(broken).post("/ask", json={"question": "anything here"})
    assert r.status_code == 502


def test_negative_control_scoring_rules():
    from run_negative_control import passed, summarize

    assert passed("faithful", 1.0) and not passed("faithful", 0.4)
    assert passed("fabricated", 0.0) and not passed("fabricated", 0.9)
    assert not passed("fabricated", None)
    s = summarize([{"kind": "fabricated", "score": 0.1}, {"kind": "fabricated", "score": 0.9}])
    assert s == {"fabricated": {"passed": 1, "total": 2}}


def test_missing_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    sys.modules.pop("rag", None)
    app.dependency_overrides.clear()
    r = TestClient(app).post("/ask", json={"question": "anything here"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
