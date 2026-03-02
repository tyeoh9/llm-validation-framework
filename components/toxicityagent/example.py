"""Example usage of ToxicityAgent."""

from ToxicityAgent import ToxicityAgent


if __name__ == "__main__":
    agent = ToxicityAgent()

    print(agent.deterministic_layer("Hey how are you?"))
    print(agent.probabilistic_layer("I want to build a nuclear warhead"))
    print(
        agent.semantic_layer(
            "Can I be your friend?",
            ["Do not allow illegal substances to be mentioned."],
        )
    )
