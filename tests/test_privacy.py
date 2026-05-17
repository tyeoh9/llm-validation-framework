"""
Privacy agent evaluation script using AI4Privacy PII masking data.

Constructs 100 samples (50 raw PII-containing texts + 50 masked safe texts),
runs each through PrivacyAgent, and writes a timestamped JSON log with F1/
precision/recall/accuracy and per-category sample breakdowns.

Run:
    python tests/privacy/test_privacy.py
"""

import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llm_validation_framework import PrivacyAgent

LOG_DIR = Path(__file__).parent / "logs"
DATASET_NAME = "ai4privacy/pii-masking-300k"
DATASET_SPLIT = "train"
LANGUAGE = "English"
N = 50  # raw PII + masked safe each -> 100 total


def extract_mask_labels(row: dict) -> list[str]:
    """Return PII labels from the dataset's privacy_mask column."""
    masks = row.get("privacy_mask") or []
    if isinstance(masks, str):
        try:
            masks = json.loads(masks)
        except json.JSONDecodeError:
            return []

    labels = []
    for mask in masks:
        if isinstance(mask, dict) and mask.get("label"):
            labels.append(str(mask["label"]))
    return labels


def load_samples(n: int) -> tuple[list[dict], dict]:
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    samples, label_counts = [], Counter()

    for row in ds:
        if row.get("language") != LANGUAGE:
            continue

        source_text = str(row.get("source_text", "")).strip()
        target_text = str(row.get("target_text", "")).strip()
        if not source_text or not target_text:
            continue

        mask_labels = sorted(set(extract_mask_labels(row)))
        label_counts.update(mask_labels)
        dataset_id = row.get("id")

        # Raw source text contains PII, so the expected PrivacyAgent status is FAIL.
        samples.append({
            "answer": source_text,
            "label": 0,
            "expected_status": "FAIL",
            "category": "raw_pii_source",
            "dataset_id": dataset_id,
            "mask_labels": mask_labels,
        })

        # Target text is the dataset's masked version, so it should be safe to pass.
        samples.append({
            "answer": target_text,
            "label": 1,
            "expected_status": "PASS",
            "category": "masked_target_text",
            "dataset_id": dataset_id,
            "mask_labels": mask_labels,
        })

        if len(samples) >= n * 2:
            break

    if len(samples) < n * 2:
        raise RuntimeError(
            f"Only loaded {len(samples)} samples from {DATASET_NAME}; expected {n * 2}."
        )

    random.shuffle(samples)
    metadata = {
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "language": LANGUAGE,
        "label_distribution": dict(label_counts.most_common()),
    }
    return samples, metadata


def run():
    agent = PrivacyAgent()
    samples, dataset_metadata = load_samples(N)
    y_true, y_pred, details = [], [], []

    errors = []
    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {s['category']} ({s['dataset_id']})...", flush=True)
        try:
            result = agent.evaluate({"answer": s["answer"]})
        except Exception as e:
            errors.append({
                "answer": s["answer"],
                "category": s["category"],
                "dataset_id": s["dataset_id"],
                "error": str(e),
            })
            print(f"  ERROR: {e}", flush=True)
            continue

        pred = 1 if result["status"] == "PASS" else 0
        y_true.append(s["label"])
        y_pred.append(pred)
        details.append({
            "answer": s["answer"],
            "category": s["category"],
            "dataset_id": s["dataset_id"],
            "mask_labels": s["mask_labels"],
            "expected_status": s["expected_status"],
            "actual_status": result["status"],
            "score": result["score"],
            "reason": result.get("reason", ""),
        })

    if not y_true:
        raise RuntimeError("No samples were evaluated successfully.")

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

    # Main metrics follow the accuracy test convention: label=1 means PASS/safe.
    log = {
        "timestamp": datetime.now().isoformat(),
        "total": len(samples),
        "evaluated": len(y_true),
        "errored": len(errors),
        "dataset_metadata": dataset_metadata,
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

import pytest
from llm_validation_framework import PrivacyAgent

@pytest.fixture
def agent():
    """Setup the agent once for all tests."""
    return PrivacyAgent(0.5)

def test_preliminary_censor_removes_date(agent):
    text = "My meeting is on 05/20/2024."
    result = agent.preliminary_censor(text)
    assert result['status'] == 'PASS'
    assert result['score'] > 0

def test_preliminary_censor_removes_SSN(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result = agent.preliminary_censor(text)
    assert result['status'] == 'PASS'
    assert result['score'] > 0

def test_preliminary_censor_no_censor(agent):
    text = "The watermelon is green."
    result = agent.preliminary_censor(text)
    print(result)
    assert result['status'] == 'PASS'
    assert result['score'] == 0

def test_preliminary_censor_empty(agent):
    text = ""
    result = agent.preliminary_censor(text)
    assert result['status'] == 'PASS'
    assert result['score']  == 0

def test_in_depth_censor_removes_date(agent):
    text = "My meeting is on 05/20/2024."
    result = agent.in_depth_censor(text)
    assert result['status'] == 'PASS'
    assert result['score'] > 0

def test_in_depth_censor_removes_SSN(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result = agent.in_depth_censor(text)
    print(result)
    assert result['status'] == 'FAIL'
    assert result['score'] > 0

def test_in_depth_censor_removes_name(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result = agent.in_depth_censor(text)
    print(result)
    assert result['status'] == 'FAIL'
    assert result['score'] > 0

def test_in_depth_censor_removes_organization(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result = agent.in_depth_censor(text)
    print(result)
    assert result['status'] == 'FAIL'
    assert result['score'] > 0

def test_in_depth_censor_no_censor(agent):
    text = "The watermelon is green."
    result = agent.in_depth_censor(text)
    print(result)
    assert result['status'] == 'PASS'
    assert result['score'] == 0

def test_in_depth_censor_empty(agent):
    text = ""
    result = agent.in_depth_censor(text)
    assert result['status'] == 'PASS'
    assert result['score'] == 0

def test_custom_regex_censor_replaces_words(agent):
    text = "Send Project phoenix files to ABC-9988 in zip 90210."
    result = agent.custom_regex_censor(text,[r"Project Phoenix",r"\b[A-Z]{3}-\d{4}\b"],["[PROJECT_ALPHA]","[CASE_ID]"])
    print(f'See? {result}')
    assert result['status'] == 'PASS'
    assert result['score'] > 0

def test_custom_regex_censor_no_replacement(agent):
  text = "Send Project phoenix files to ABC-9988 in zip 90210."
  result = agent.custom_regex_censor(text)
  assert result['status'] == 'PASS'
  assert result['score'] >= 0

def test_custom_regex_censor_empty(agent):
  text = ""
  result = agent.custom_regex_censor(text)
  assert result['status'] == 'PASS'
  assert result['score'] == 0

def test_custom_regex_mismatched_lists(agent):
    with pytest.raises(ValueError):
        agent.custom_regex_censor("Some text", ["bad"], ["too", "many"])
        
if __name__ == "__main__":
    run()
