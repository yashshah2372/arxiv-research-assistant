# ArXiv Research Assistant — Agentic RAG with an Evaluation Harness

Answers research questions over a corpus of arXiv papers on retrieval-augmented generation, with **grounded, cited** answers. It compares two architectures on the same questions, a single-shot RAG pipeline and a LangGraph ReAct agent, and measures both with a small evaluation harness instead of eyeballing outputs.

## What it does
- Ingests 25 recent arXiv RAG papers, chunks and embeds them into a Chroma vector store.
- **Single-shot RAG** (`rag.py`): retrieves the top-k papers and generates an answer grounded only in that context, with citations.
- **Agentic RAG** (`agent.py`): a LangGraph ReAct agent with a `search_papers` tool that can search several times for cross-paper questions.
- Says "I don't have enough information" when the corpus doesn't cover the question (anti-hallucination guardrail).

## Architecture
```
Ingestion:  arXiv API → papers.json → chunk → embed → Chroma (vectors + metadata)
Single-shot: question → retrieve top-k → grounded generation with citations
Agentic:     question → ReAct agent ⇄ search_papers tool (one call per topic) → cited answer
Evaluation:  golden set → recall@k (retrieval) + LLM-as-judge faithfulness (generation)
```

## Evaluation
`eval/golden_set.json` holds 10 hand-written questions: 6 answerable from one or more papers, 2 partially answerable, and 2 the corpus can't answer (to test abstention). Each lists the relevant paper IDs and the expected behaviour.

| Script | Measures |
|---|---|
| `eval/run_eval.py` | **Retrieval recall@k** for k = 1, 3, 5, 10 over the 8 questions that have relevant papers. Abstention questions are reported separately so they can't inflate the mean. |
| `eval/run_faithfulness.py` | **Faithfulness of the single-shot pipeline**: a judge LLM checks whether every claim in the answer is supported by the retrieved context. It never sees the expected answer, so it measures grounding, not correctness. A correct abstention counts as faithful. |
| `eval/run_faithfulness_agent.py` | The same faithfulness check on the **agent path**, including tool-call failures. |
| `eval/run_negative_control.py` | **Checks the judge itself.** Feeds it answers with known faithfulness (faithful, fabricated, correct refusal) built from real retrieved context, and fails if it can't tell them apart. A judge that rates everything as faithful would make the scores above meaningless. |

### Results (last run)
| Metric | Result |
|---|---|
| Retrieval recall@k | 56% at k=1 → 84% at k=10 |
| Faithfulness, single-shot | 1.00 |
| Faithfulness, agent | 0.86 |
| Agent tool-call failure rate | 20% |

The takeaway: on a small, focused corpus the simpler pipeline was more faithful than the agent. The agent's extra reasoning steps gave it more room to drift from the retrieved context, and its tool calls sometimes failed. More autonomy is not automatically better; measure it.

## API

A FastAPI service wraps the single-shot pipeline:

```bash
docker build -t arxiv-rag .          # also builds the vector DB at build time
docker run -p 8000:8000 -e GROQ_API_KEY=your_key arxiv-rag

curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How do papers evaluate RAG systems?", "k": 3}'
```

Returns `{"answer": "... [1]", "sources": [{"number": 1, "title": "...", "url": "..."}]}`. `GET /health` is used for container checks. Offline tests (`pytest -q`) cover the API with a stubbed pipeline, and CI builds the image and checks that the container starts.

## Stack
- Python
- **Chroma** (local vector DB, MiniLM embeddings)
- **Groq** (Llama 3.3 70B through an OpenAI-compatible API); provider-agnostic, swap `base_url` for OpenAI or Claude
- **LangGraph** (ReAct agent orchestration)
- **FastAPI** + **Docker** (API, with the vector DB built at image build time)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# create a .env file with your Groq key:
echo "GROQ_API_KEY=your_key_here" > .env

python ingest.py                # builds the vector DB from papers.json
python rag.py                   # single-shot: ask a question, get a cited answer
python agent.py                 # agentic version

python eval/run_eval.py                    # retrieval recall@k
python eval/run_faithfulness.py            # faithfulness, single-shot
python eval/run_faithfulness_agent.py      # faithfulness, agent
python eval/run_negative_control.py        # does the judge catch fabricated answers?
```

## Status
- [x] Ingestion pipeline (fetch, chunk, embed, store)
- [x] End-to-end RAG with citations
- [x] Agentic layer (LangGraph ReAct agent with a search tool)
- [x] Evaluation harness (recall@k, LLM-as-judge faithfulness)
- [x] Serve as an API (FastAPI) and package with Docker
