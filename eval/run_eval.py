import json
from pathlib import Path

import chromadb

HERE = Path(__file__).parent      # .../eval
ROOT = HERE.parent                # repo root - chroma_db lives here

K_VALUES = [1, 3, 5, 10]
DETAIL_K = 5                      # which k to print the per-question table for


def recall_at_k(retrieved_ids, relevant_ids):
    """Fraction of the relevant documents that made it into the top k."""
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)


def main():
    with open(HERE / "golden_set.json") as f:
        golden = json.load(f)

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection("rag_papers")

    # Questions with no relevant documents have an undefined recall (0/0) and
    # test abstention - a generation behaviour - rather than retrieval. They are
    # reported separately instead of being folded into the mean, which would
    # otherwise inflate the headline number with questions retrieval can't fail.
    scored = [q for q in golden if q["relevant_doc_ids"]]
    abstain = [q for q in golden if not q["relevant_doc_ids"]]

    print(f"corpus: {collection.count()} papers   "
          f"scored: {len(scored)}   abstention: {len(abstain)}\n")

    curve = {}
    for k in K_VALUES:
        rows = []
        for q in scored:
            retrieved = collection.query(
                query_texts=[q["question"]], n_results=k
            )["ids"][0]
            score = recall_at_k(retrieved, q["relevant_doc_ids"])
            missed = set(q["relevant_doc_ids"]) - set(retrieved)
            rows.append((q, score, missed))

        curve[k] = sum(score for _, score, _ in rows) / len(rows)

        if k == DETAIL_K:
            print(f"per-question recall@{k}")
            print("-" * 68)
            for q, score, missed in rows:
                flag = "   " if score == 1.0 else "<- "
                print(f"{flag}{q['id']:<4} {score:>5.2f}   {q['question'][:46]}")
                if missed:
                    print(f"          missed: {', '.join(sorted(missed))}")
            print()

    print("recall@k curve")
    print("-" * 68)
    for k in K_VALUES:
        print(f"   k={k:<3} {curve[k]:.3f}   {'#' * int(curve[k] * 40)}")

    print("\nabstention cases - excluded from recall, checked in the "
          "faithfulness pass:")
    for q in abstain:
        print(f"   {q['id']}   {q['question'][:58]}")


if __name__ == "__main__":
    main()
