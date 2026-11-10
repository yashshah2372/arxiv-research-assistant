"""Negative control for the faithfulness judge.

A judge that rates everything as faithful would make the faithfulness numbers
meaningless. This script feeds the judge answers whose faithfulness is known in
advance and checks that it tells them apart:

- faithful:   only restates what the context says       -> expect a high score
- fabricated: invents claims the context never makes    -> expect a low score
- abstain:    declines when the context lacks the answer -> expect a high score

The cases are built from real retrieved context for golden-set questions, so the
judge sees the same kind of input as in run_faithfulness.py.

Run:  ./venv/bin/python eval/run_negative_control.py
"""

import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

load_dotenv(ROOT / ".env")

from run_faithfulness import TOP_K, build_context, judge  # noqa: E402

PASS_HIGH = 0.8   # faithful and abstain answers should score at least this
PASS_LOW = 0.5    # fabricated answers should score below this

# Claims that appear in none of the indexed abstracts.
FABRICATED = (
    "This approach was first introduced by Google DeepMind in 2015 and requires "
    "a quantum computer to run. The paper reports 100% accuracy on every benchmark "
    "and shows that retrieval makes models slower without improving answers."
)


def build_cases(collection, golden):
    cases = []
    for q in golden:
        if not q["relevant_doc_ids"]:
            continue
        res = collection.query(query_texts=[q["question"]], n_results=TOP_K)
        context = build_context(res["documents"][0], res["metadatas"][0])
        first_doc = res["documents"][0][0]
        # Faithful: quote the first retrieved abstract back, attributed to source [1].
        faithful = f"According to [1], {first_doc.split('. ')[0].strip().rstrip('.')}."
        cases.append({"id": q["id"], "kind": "faithful", "context": context, "answer": faithful})
        cases.append({"id": q["id"], "kind": "fabricated", "context": context, "answer": FABRICATED})
        cases.append({"id": q["id"], "kind": "abstain", "context": context,
                      "answer": "I don't have enough information in the provided context to answer that."})
    return cases


def passed(kind, score):
    if score is None:
        return False
    return score < PASS_LOW if kind == "fabricated" else score >= PASS_HIGH


def summarize(results):
    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r)
    return {
        kind: {"passed": sum(passed(kind, r["score"]) for r in rs), "total": len(rs)}
        for kind, rs in by_kind.items()
    }


def main():
    with open(HERE / "golden_set.json") as f:
        golden = json.load(f)
    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("rag_papers")
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

    results = []
    for case in build_cases(collection, golden):
        score, unsupported = judge(client, case["context"], case["answer"])
        results.append({**case, "score": score})
        mark = "ok " if passed(case["kind"], score) else "FAIL"
        shown = "n/a" if score is None else f"{score:.2f}"
        print(f"{mark} {case['id']:<4} {case['kind']:<10} score={shown}")

    print("-" * 40)
    summary = summarize(results)
    for kind, s in summary.items():
        print(f"{kind:<10} {s['passed']}/{s['total']} as expected")
    return 0 if all(s["passed"] == s["total"] for s in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
