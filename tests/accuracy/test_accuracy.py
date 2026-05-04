"""
Accuracy agent evaluation script using TriviaQA.

Constructs 100 samples (50 correct + 50 shuffled-wrong answers), runs each
through AccuracyAgent, and writes a timestamped JSON log with F1/precision/
recall/accuracy and per-category sample breakdowns.

Run:
    python tests/accuracy/test_accuracy.py
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from components.accuracy.AccuracyAgent import AccuracyAgent

LOG_DIR = Path(__file__).parent / "logs"
N = 50  # positive + negative each → 100 total


def load_samples(n: int) -> list[dict]:
    ds = load_dataset("trivia_qa", "rc.web", split="validation", trust_remote_code=True)
    questions, answers = [], []
    for row in ds:
        questions.append(row["question"])
        answers.append(row["answer"]["value"])
        if len(questions) >= n:
            break

    positives = [{"question": q, "answer": a, "label": 1} for q, a in zip(questions, answers)]
    # Shift answers by 1 to create plausible-but-wrong negatives
    negatives = [{"question": q, "answer": answers[(i + 1) % n], "label": 0}
                 for i, q in enumerate(questions)]
    samples = positives + negatives
    random.shuffle(samples)
    return samples


def run():
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
    print(f"TP={len(buckets['true_positives'])}  FP={len(buckets['false_positives'])}  "
          f"TN={len(buckets['true_negatives'])}  FN={len(buckets['false_negatives'])}  "
          f"errors={len(errors)}")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    run()
