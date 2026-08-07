"""Faithfulness evaluation for the arXiv research assistant.

For each golden question: retrieve -> generate an answer -> ask a judge LLM
whether every claim in that answer is supported by the retrieved context.

Faithfulness measures GROUNDING, not correctness. The judge never sees the
expected answer - only the context the generator actually had. An answer can be
factually true and still unfaithful (answered from the model's own knowledge),
which is exactly the failure this catches.

Correctly abstaining ("I don't have enough information") counts as faithful.

Run:  ./venv/bin/python eval/run_faithfulness.py
"""

import json
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parent

TOP_K = 5
MODEL = "llama-3.3-70b-versatile"

load_dotenv(ROOT / ".env")


def build_context(docs, metas):
    return "\n\n".join(
        f"[{i}] {m['title']}\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas), 1)
    )


def generate_answer(client, question, context):
    prompt = f"""Answer the question using ONLY the context below.
Cite the sources you use with their [number]. If the context does not
contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


JUDGE_PROMPT = """You are grading an answer for FAITHFULNESS to its source context.

Faithfulness means: every factual claim in the answer is supported by the
context below. It does NOT mean the answer is correct, complete, or useful.

Rules:
- A claim stated in the answer but absent from the context is UNSUPPORTED,
  even if it happens to be true in the real world.
- If the answer declines to answer ("I don't have enough information") and the
  context genuinely lacks that information, that is FULLY FAITHFUL - score 1.0.
- Attributing a topic to a paper that does not discuss it is UNSUPPORTED.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a JSON object, no prose, no code fences:
{{"score": <float 0.0-1.0, fraction of claims that are supported>,
  "unsupported": ["<each unsupported claim, briefly>"]}}"""


def judge(client, context, answer):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(context=context, answer=answer),
        }],
        temperature=0,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.S)          # tolerate stray prose
    if not match:
        return None, [f"judge returned unparseable output: {raw[:80]}"]
    try:
        verdict = json.loads(match.group(0))
        return float(verdict["score"]), verdict.get("unsupported", [])
    except (ValueError, KeyError) as exc:
        return None, [f"judge JSON invalid ({exc}): {raw[:80]}"]


def main():
    with open(HERE / "golden_set.json") as f:
        golden = json.load(f)

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection("rag_papers")

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    print(f"faithfulness @ k={TOP_K}   judge={MODEL}\n")
    print("-" * 72)

    scores, by_behavior = [], {}
    for q in golden:
        res = collection.query(query_texts=[q["question"]], n_results=TOP_K)
        context = build_context(res["documents"][0], res["metadatas"][0])

        answer = generate_answer(client, q["question"], context)
        score, unsupported = judge(client, context, answer)

        if score is not None:
            scores.append(score)
            by_behavior.setdefault(q["expected_behavior"], []).append(score)

        shown = "  n/a" if score is None else f"{score:>5.2f}"
        flag = "   " if score == 1.0 else "<- "
        print(f"{flag}{q['id']:<4}{shown}  [{q['expected_behavior']:<7}] "
              f"{q['question'][:40]}")
        print(f"          answer: {answer[:96].replace(chr(10), ' ')}...")
        for claim in unsupported[:3]:
            print(f"          UNSUPPORTED: {claim[:80]}")
        print()

    print("-" * 72)
    print(f"mean faithfulness: {sum(scores) / len(scores):.3f}  "
          f"({len(scores)}/{len(golden)} scored)")
    for behavior, vals in sorted(by_behavior.items()):
        print(f"   {behavior:<8} {sum(vals) / len(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
