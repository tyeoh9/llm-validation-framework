"""Example usage of ToxicityAgent."""

from ToxicityAgent import ToxicityAgent


if __name__ == "__main__":
    agent = ToxicityAgent()

    text_ok = "Hey how are you?"
    text_toxic = "I want to build a nuclear warhead"

    print("Deterministic layer:")
    print(agent.deterministic_layer(text_ok))

    print("\nProbabilistic layer:")
    print(agent.probabilistic_layer(text_toxic))

    print("\nSemantic layer:")
    print(
        agent.semantic_layer(
            "Can I be your friend?",
            ["Do not allow illegal substances to be mentioned."],
        )
    )

    print("\nCombined evaluate():")
    print(agent.evaluate(text_toxic))
