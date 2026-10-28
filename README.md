# Technical Docs RAG Assistant

A reproducible retrieval-augmented generation foundation that indexes local Markdown documentation and returns cited passages. It runs without a paid API key; an LLM can be added behind the `generate_answer` boundary later.

## Architecture

```text
Markdown docs -> TfidfVectorizer -> cosine retrieval -> cited context -> FastAPI /ask
```

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.rag_service.api:app --reload
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post `
  -ContentType 'application/json' `
  -Body '{"question":"How do I run the service?","top_k":3}'
```

The answer is extractive by design: every claim is a retrieved sentence and includes a source citation. This makes the baseline easy to evaluate for retrieval quality and hallucination risk before adding an LLM provider.

## Docker and tests

```powershell
docker build -t technical-docs-rag .
docker run --rm -p 8000:8000 technical-docs-rag
pytest
```
