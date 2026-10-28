from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .retriever import Retriever

app = FastAPI(title="Technical Docs RAG Assistant", version="1.0.0")
retriever = Retriever(Path(__file__).parents[2] / "docs")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class Citation(BaseModel):
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        results = retriever.search(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    citations = [
        Citation(source=chunk.source, score=round(score, 4), text=chunk.text)
        for chunk, score in results
        if score > 0
    ]
    answer = (
        " ".join(citation.text for citation in citations)
        if citations
        else "I could not find supporting information in the indexed documentation."
    )
    return AskResponse(answer=answer, citations=citations)
