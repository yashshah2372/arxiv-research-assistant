# ArXiv Research Assistant — Agentic RAG

An agentic RAG system that answers research questions over a corpus of arXiv papers on retrieval-augmented generation, with **grounded, cited** answers.

## What it does
- Ingests recent arXiv RAG papers, chunks + embeds them into a vector store.
- Retrieves the most relevant papers for a question and generates an answer **grounded only in the retrieved context**, citing sources.
- Says "I don't have enough information" when the corpus doesn't cover the question (anti-hallucination guardrail).

## Architecture
```
Ingestion:  arXiv API → papers.json → chunk → embed → Chroma (vectors + metadata)
Serving:    question → retrieve top-k → grounded generation with citations
Agentic:    LangGraph ReAct agent + search tool → multi-step / cross-paper reasoning   (in progress)
```

## Stack
- Python
- **Chroma** (local vector DB, MiniLM embeddings)
- **Groq** (Llama 3.3, OpenAI-compatible API) — provider-agnostic (swap `base_url` for OpenAI/Claude)
- **LangGraph** (agent orchestration)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# create a .env file with your Groq key:
echo "GROQ_API_KEY=your_key_here" > .env

python ingest.py                # builds the vector DB from papers.json
python rag.py                   # ask a question, get a cited answer
```

## Status
- [x] Ingestion pipeline (fetch, chunk, embed, store)
- [x] End-to-end RAG with citations
- [ ] Agentic layer (LangGraph, multi-step)
- [ ] Evaluation harness (recall@k, faithfulness, LLM-as-judge)
- [ ] Deployment (FastAPI + Docker)
