"""Example usage of AccuracyAgent for text equivalence evaluation."""

from llm_validation_framework import AccuracyAgent

if __name__ == "__main__":
    agent = AccuracyAgent()

    # Example 1: True fact - should PASS
    result = agent.evaluate({
        "question": "Where is the Eiffel Tower?",
        "answer": "The Eiffel Tower is located in Paris, France.",
    })
    print(f"Status: {result['status']}")
    print(f"Score: {result['score']}")
    print(f"Reason: {result['reason']}\n")

    # Example 2: Common misconception - should FAIL
    result2 = agent.evaluate({
        "question": "Is the Great Wall visible from space?",
        "answer": "The Great Wall of China is visible from space with the naked eye.",
    })
    print(f"Status: {result2['status']}")
    print(f"Score: {result2['score']}")
    print(f"Reason: {result2['reason']}\n")
