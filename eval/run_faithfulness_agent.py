"""Faithfulness evaluation of the AGENT path (agent.py).

Same golden set and same judge as run_faithfulness.py, but the answer comes
from the LangGraph ReAct agent instead of the single-shot rag.py pipeline.

The difference that matters: the agent chooses its own search queries and may
issue several, so the context it actually saw is the union of every tool result
it received - not one top-k slice. The judge grades the answer against exactly
that union, which is the only fair test of whether the agent invented anything.

Run:  ./venv/bin/python eval/run_faithfulness_agent.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parent

sys.path.insert(0, str(ROOT))       # import agent.py from the repo root
sys.path.insert(0, str(HERE))       # reuse the judge from the sibling script

load_dotenv(ROOT / ".env")

from run_faithfulness import judge, MODEL      # noqa: E402
import agent as agent_module                   # noqa: E402


def run_agent(question):
    """Return (final_answer, context_the_agent_saw, tool_call_count)."""
    result = agent_module.agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = result["messages"]

    tool_outputs, calls = [], 0
    for m in messages:
        if m.__class__.__name__ == "ToolMessage":
            tool_outputs.append(m.content)
        calls += len(getattr(m, "tool_calls", []) or [])

    final = messages[-1].content
    return final, "\n\n".join(tool_outputs), calls


def main():
    with open(HERE / "golden_set.json") as f:
        golden = json.load(f)

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    print(f"faithfulness of AGENT path   TOP_K={agent_module.TOP_K}   "
          f"judge={MODEL}\n")
    print("-" * 72)

    scores, by_behavior = [], {}
    for q in golden:
        try:
            answer, context, calls = run_agent(q["question"])
        except Exception as exc:                       # agent loops can fail
            print(f"   {q['id']:<4}  ERR  {type(exc).__name__}: {exc}"[:100])
            print()
            continue

        if not context:
            # The agent answered without calling the tool at all - nothing was
            # retrieved, so every factual claim is ungrounded by construction.
            score, unsupported = 0.0, ["agent answered without retrieving"]
        else:
            score, unsupported = judge(client, context, answer)

        if score is not None:
            scores.append(score)
            by_behavior.setdefault(q["expected_behavior"], []).append(score)

        shown = "  n/a" if score is None else f"{score:>5.2f}"
        flag = "   " if score == 1.0 else "<- "
        print(f"{flag}{q['id']:<4}{shown}  [{q['expected_behavior']:<7}] "
              f"{calls} tool call(s)  {q['question'][:34]}")
        print(f"          answer: {answer[:96].replace(chr(10), ' ')}...")
        for claim in unsupported[:3]:
            print(f"          UNSUPPORTED: {claim[:80]}")
        print()

    print("-" * 72)
    if scores:
        print(f"mean faithfulness (agent): {sum(scores) / len(scores):.3f}  "
              f"({len(scores)}/{len(golden)} scored)")
        for behavior, vals in sorted(by_behavior.items()):
            print(f"   {behavior:<8} {sum(vals) / len(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
