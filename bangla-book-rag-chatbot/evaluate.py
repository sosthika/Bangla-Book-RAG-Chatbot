
"""Compare retrieval hit rate for the two chunking strategies."""
import json
from pathlib import Path

from rag_pipeline import get_vectorstore

TEST_FILE = Path("tests/test_questions.json")
TOP_K = 4


def evaluate_strategy(strategy, questions):
    store = get_vectorstore(strategy)
    total = hits = 0
    details = []

    for item in questions:
        if not item.get("in_book"):
            continue

        total += 1
        expected = item["expected_source"]
        retrieved = store.similarity_search(item["question"], k=TOP_K)

        retrieved_pairs = [
            (doc.metadata.get("section"), doc.metadata.get("chapter"))
            for doc in retrieved
        ]
        expected_pair = (expected["section"], expected["chapter"])
        hit = expected_pair in retrieved_pairs
        hits += int(hit)

        details.append({
            "id": item["id"],
            "question": item["question"],
            "expected_source": expected_pair,
            "retrieved_sources": retrieved_pairs,
            "hit": hit,
        })
        print(f"{'HIT ' if hit else 'MISS'} | Q{item['id']} | {item['question']}")

    rate = (hits / total * 100) if total else 0.0
    return {
        "strategy": strategy,
        "total_in_book_questions": total,
        "hits": hits,
        "hit_rate_percent": round(rate, 2),
        "top_k": TOP_K,
        "details": details,
    }


def main():
    questions = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    results = {
        "strategy_a": evaluate_strategy("strategy_a", questions),
        "strategy_b": evaluate_strategy("strategy_b", questions),
        "metric": "A hit means at least one of top-k retrieved chunks belongs to the expected section/chapter.",
    }

    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== CHUNKING COMPARISON ===")
    print("Strategy A:", results["strategy_a"]["hit_rate_percent"], "%")
    print("Strategy B:", results["strategy_b"]["hit_rate_percent"], "%")
    print("Saved: evaluation/results.json")
    print("Use these measured results in your report; do not invent scores.")


if __name__ == "__main__":
    main()
