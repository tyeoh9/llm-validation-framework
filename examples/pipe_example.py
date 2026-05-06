"""Example: run validation pipeline on LLM input + output."""

from llm_validation_framework import Pipe, ToxicityAgent, PrivacyAgent, AccuracyAgent


def main():
    toxicity_agent = ToxicityAgent()
    privacy_agent = PrivacyAgent()
    accuracy_agent = AccuracyAgent(config_path=None)

    pipe = Pipe(steps=[toxicity_agent, privacy_agent, accuracy_agent])

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

        results = pipe.evaluate({"question": question, "answer": answer})

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
