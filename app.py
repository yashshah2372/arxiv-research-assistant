"""HTTP API for the single-shot RAG pipeline.

Run locally:  uvicorn app:app --reload
Docker:       docker build -t arxiv-rag . && docker run -p 8000:8000 -e GROQ_API_KEY=... arxiv-rag
"""

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ArXiv Research Assistant", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    number: int
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def get_answerer():
    # Imported lazily so the app (and its tests) start without a vector DB or API key.
    try:
        from rag import answer
    except KeyError as exc:  # rag.py reads GROQ_API_KEY at import time
        raise HTTPException(status_code=503, detail=f"service not configured: missing {exc.args[0]}")
    return answer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, answer=Depends(get_answerer)):
    try:
        text, metas = answer(req.question, k=req.k)
    except Exception as exc:  # upstream LLM or vector DB failure
        raise HTTPException(status_code=502, detail=f"answer generation failed: {type(exc).__name__}")
    sources = [Source(number=i, title=m["title"], url=m.get("url", "")) for i, m in enumerate(metas, 1)]
    return AskResponse(answer=text, sources=sources)
