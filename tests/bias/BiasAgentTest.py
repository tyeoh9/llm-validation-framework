import json
from collections import defaultdict

from BiasAgent import BiasAgent


# ----------------------------
# Config
# ----------------------------
MANUAL_DATA_FILE = "sbic_bias_agent_test_200_manual_pass_fail.json"
RESULTS_FILE = "sbic_bias_agent_manual_results.json"


# ----------------------------
# Load manual labeled data
# ----------------------------
def load_manual_data(path):
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


# ----------------------------
# Run evaluation
# ----------------------------
def run_bias_agent(test_set):
    print("\nInitializing agent...")
    agent = BiasAgent()  

    results = []
    total = len(test_set)

    for idx, item in enumerate(test_set, start=1):
        question = item.get("question", "Please answer the question.")
        answer = item["answer"]

        try:
            out = agent.evaluate({
                "question": question,
                "answer": answer,
            })

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

            # Optional extra fields if using BiasAgentV2
            if "target_group_present" in out:
                record["target_group_present"] = out.get("target_group_present")
            if "target_group" in out:
                record["target_group"] = out.get("target_group")
            if "bias_type" in out:
                record["bias_type"] = out.get("bias_type")
            if "confidence" in out:
                record["confidence"] = out.get("confidence")
            if "subscores" in out:
                record["subscores"] = out.get("subscores", {})

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

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {RESULTS_FILE}")
    return results


# ----------------------------
# Summary
# ----------------------------
def summarize_results(results):
    valid = [r for r in results if r["predicted"] in {"PASS", "FAIL"}]

    if not valid:
        print("\nNo valid PASS/FAIL results to summarize.")
        return

    total = len(valid)
    correct = sum(r["correct"] for r in valid)
    accuracy = correct / total if total else 0.0

    tp = sum(1 for r in valid if r["manual_expected"] == "FAIL" and r["predicted"] == "FAIL")
    tn = sum(1 for r in valid if r["manual_expected"] == "PASS" and r["predicted"] == "PASS")
    fp = sum(1 for r in valid if r["manual_expected"] == "PASS" and r["predicted"] == "FAIL")
    fn = sum(1 for r in valid if r["manual_expected"] == "FAIL" and r["predicted"] == "PASS")

    precision_fail = tp / (tp + fp) if (tp + fp) else 0.0
    recall_fail = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    avg_score = sum(r["score"] for r in valid if r["score"] is not None) / total

    print("\n========== SUMMARY ==========")
    print(f"Total valid cases:      {total}")
    print(f"Accuracy:               {accuracy:.2%}")
    print(f"FAIL precision:         {precision_fail:.2%}")
    print(f"FAIL recall:            {recall_fail:.2%}")
    print(f"False positive rate:    {false_positive_rate:.2%}")
    print(f"False negative rate:    {false_negative_rate:.2%}")
    print(f"Average score:          {avg_score:.4f}")

    by_category = defaultdict(lambda: {"n": 0, "fail_pred": 0, "score_sum": 0.0})
    for r in valid:
        cat = r["category"]
        by_category[cat]["n"] += 1
        by_category[cat]["fail_pred"] += int(r["predicted"] == "FAIL")
        by_category[cat]["score_sum"] += r["score"] if r["score"] is not None else 0.0

    print("\nFail rate by category:")
    for cat, stats in sorted(by_category.items()):
        n = stats["n"]
        fail_rate = stats["fail_pred"] / n if n else 0.0
        avg = stats["score_sum"] / n if n else 0.0
        print(f"- {cat}: n={n}, fail_rate={fail_rate:.2%}, avg_score={avg:.4f}")

    print("\nSample false positives (manual PASS, predicted FAIL):")
    false_positives = [
        r for r in valid
        if r["manual_expected"] == "PASS" and r["predicted"] == "FAIL"
    ][:5]
    for r in false_positives:
        print(f"\nID: {r['id']}")
        print(f"Category: {r['category']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Reason: {r['reason']}")
        print(f"Text: {r['answer'][:300]}")

    print("\nSample false negatives (manual FAIL, predicted PASS):")
    false_negatives = [
        r for r in valid
        if r["manual_expected"] == "FAIL" and r["predicted"] == "PASS"
    ][:5]
    for r in false_negatives:
        print(f"\nID: {r['id']}")
        print(f"Category: {r['category']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Reason: {r['reason']}")
        print(f"Text: {r['answer'][:300]}")


# ----------------------------
# Compare original vs manual labels
# ----------------------------
def summarize_label_changes(test_set):
    changed = [
        item for item in test_set
        if item.get("expected") != item.get("manual_expected")
    ]

    print("\n========== LABEL CHANGES ==========")
    print(f"Total changed from original SBIC-derived label: {len(changed)}")

    if changed:
        print("\nSample changed labels:")
        for item in changed[:10]:
            print(
                f"- {item['id']}: original={item.get('expected')} "
                f"manual={item.get('manual_expected')} | {item['answer'][:120]}"
            )


# ----------------------------
# Main
# ----------------------------
def main():
    test_set = load_manual_data(MANUAL_DATA_FILE)
    summarize_label_changes(test_set)
    results = run_bias_agent(test_set)
    summarize_results(results)


if __name__ == "__main__":
    main()