"""Example: run validation pipeline on LLM input + output (toxicity then accuracy)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.pipe.Pipe import Pipe
from components.toxicityagent.ToxicityAgent import ToxicityAgent
from components.accuracy.AccuracyAgent import AccuracyAgent


def main():
    toxicity_agent = ToxicityAgent()
    accuracy_agent = AccuracyAgent(config_path=None)

    pipe = Pipe(steps=[toxicity_agent, accuracy_agent])

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

        # For now, the pipeline operates only on the LLM answer text.
        results = pipe.evaluate(answer)

        print("\n" + "=" * 60)
        for i, (step, r) in enumerate(zip(pipe.steps, results), start=1):
            step_name = getattr(step, "name", step.__class__.__name__)
            print(
                f"Step {i} ({step_name}): "
                f"{r.get('status', '?')} | score={r.get('score', 0):.2f}"
            )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()