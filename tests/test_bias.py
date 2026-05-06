"""Bias agent evaluation using manually labeled data.

Expects a JSON file at the path in MANUAL_DATA_FILE with entries containing:
  - id, answer, manual_expected (PASS|FAIL)
  - optional: question, source, category, expected

Run:
    pytest tests/test_bias.py -s
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from llm_validation_framework import BiasAgent

MANUAL_DATA_FILE = Path(__file__).parent / "bias" / "sbic_bias_agent_test_200_manual_pass_fail.json"
RESULTS_FILE = Path(__file__).parent / "bias" / "sbic_bias_agent_manual_results.json"


def load_manual_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Manual data file must contain a JSON list.")

    required_fields = {"id", "answer", "manual_expected"}
    for i, item in enumerate(data):
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"Item at index {i} is missing required fields: {sorted(missing)}"
            )

    print(f"Loaded {len(data)} manually labeled examples from {path}")
    return data


@pytest.mark.slow
def test_bias_agent_accuracy():
    if not MANUAL_DATA_FILE.exists():
        pytest.skip(f"Manual data file not found: {MANUAL_DATA_FILE}")

    test_set = load_manual_data(MANUAL_DATA_FILE)
    agent = BiasAgent()
    results = []
    total = len(test_set)

    for idx, item in enumerate(test_set, start=1):
        question = item.get("question", "Please answer the question.")
        answer = item["answer"]

        try:
            out = agent.evaluate({"question": question, "answer": answer})
            record = {
                "id": item["id"],
                "source": item.get("source", ""),
                "category": item.get("category", ""),
                "original_expected": item.get("expected", ""),
                "manual_expected": item["manual_expected"],
                "predicted": out["status"],
                "score": out["score"],
                "reason": out["reason"],
                "answer": answer,
                "correct": out["status"] == item["manual_expected"],
            }
            results.append(record)
            print(
                f"[{idx}/{total}] {item['id']} | "
                f"manual={item['manual_expected']} predicted={out['status']} "
                f"score={out['score']:.4f}"
            )
        except Exception as e:
            results.append({
                "id": item["id"],
                "source": item.get("source", ""),
                "category": item.get("category", ""),
                "original_expected": item.get("expected", ""),
                "manual_expected": item["manual_expected"],
                "predicted": "ERROR",
                "score": None,
                "reason": str(e),
                "answer": answer,
                "correct": False,
            })
            print(f"[{idx}/{total}] {item['id']} | ERROR: {e}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid = [r for r in results if r["predicted"] in {"PASS", "FAIL"}]
    total_valid = len(valid)
    correct = sum(r["correct"] for r in valid)
    accuracy = correct / total_valid if total_valid else 0.0

    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total_valid})")
    assert accuracy >= 0.5, f"Bias agent accuracy {accuracy:.2%} is below 50%"
