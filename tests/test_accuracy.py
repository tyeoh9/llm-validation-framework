"""
Accuracy agent evaluation script using TriviaQA.

Constructs 100 samples (50 correct + 50 shuffled-wrong answers), runs each
through AccuracyAgent, and writes a timestamped JSON log with F1/precision/
recall/accuracy and per-category sample breakdowns.

Run:
    pytest tests/test_accuracy.py -s
"""

import json
import random
from datetime import datetime
from pathlib import Path

import pytest

from llm_validation_framework import AccuracyAgent

LOG_DIR = Path(__file__).parent / "logs" / "accuracy"
N = 50  # positive + negative each → 100 total


def load_samples(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.web", split="validation", trust_remote_code=True)
    questions, answers = [], []
    for row in ds:
        questions.append(row["question"])
        answers.append(row["answer"]["value"])
        if len(questions) >= n:
            break

    positives = [{"question": q, "answer": a, "label": 1} for q, a in zip(questions, answers)]
    negatives = [{"question": q, "answer": answers[(i + 1) % n], "label": 0}
                 for i, q in enumerate(questions)]
    samples = positives + negatives
    random.shuffle(samples)
    return samples


@pytest.mark.slow
def test_accuracy_agent_f1():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    agent = AccuracyAgent()
    samples = load_samples(N)
    y_true, y_pred, details = [], [], []
    errors = []

    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {s['question'][:60]}...", flush=True)
        try:
            result = agent.evaluate({"question": s["question"], "answer": s["answer"]})
        except Exception as e:
            errors.append({"question": s["question"], "answer": s["answer"], "error": str(e)})
            print(f"  ERROR: {e}", flush=True)
            continue
        pred = 1 if result["status"] == "PASS" else 0
        y_true.append(s["label"])
        y_pred.append(pred)
        details.append({
            "question": s["question"],
            "answer": s["answer"],
            "score": result["score"],
            "reason": result.get("reason", ""),
        })

    buckets = {"true_positives": [], "false_positives": [], "true_negatives": [], "false_negatives": []}
    for d, yt, yp in zip(details, y_true, y_pred):
        if yt == 1 and yp == 1:
            buckets["true_positives"].append(d)
        elif yt == 0 and yp == 1:
            buckets["false_positives"].append(d)
        elif yt == 0 and yp == 0:
            buckets["true_negatives"].append(d)
        else:
            buckets["false_negatives"].append(d)

    log = {
        "timestamp": datetime.now().isoformat(),
        "total": len(samples),
        "evaluated": len(y_true),
        "errored": len(errors),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        **buckets,
        "errors": errors,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(log, indent=2))

    print(f"\naccuracy={log['accuracy']}  precision={log['precision']}  recall={log['recall']}  f1={log['f1']}")
    print(f"Log: {log_file}")

    assert log["f1"] >= 0.5, f"F1 score {log['f1']} is below 0.5"
