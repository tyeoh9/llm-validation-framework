"""Example: run validation pipeline on LLM input + output (toxicity then accuracy)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.pipe.Pipe import Pipe
from components.pipe.steps.toxicity_layer import ToxicityStep
from components.pipe.steps.accuracy_layer import AccuracyStep


def main():
    toxicity_step = ToxicityStep(
        illegal_categories=["Do not allow illegal substances to be mentioned."],
        semantic_threshold=0.5,
    )
    accuracy_step = AccuracyStep(rag_k=5, config_path=None)

    pipe = Pipe(steps=[toxicity_step, accuracy_step])

    print("Validation pipeline: enter LLM input (question) and LLM output (answer).")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("LLM input (question): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = input("LLM output (answer): ").strip()
        if not answer:
            print("Skipping (empty answer).\n")
            continue

        payload = {"question": question, "answer": answer}
        results = pipe.evaluate(payload)

        print("\n" + "=" * 60)
        for i, r in enumerate(results, start=1):
            print(f"Step {i}: {r.get('status', '?')} | score={r.get('score', 0):.2f}")
            if r.get("reason"):
                print(f"  reason: {r['reason']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()