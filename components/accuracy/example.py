"""Example usage of AccuracyAgent for text equivalence evaluation."""

from AccuracyAgent import AccuracyAgent

if __name__ == "__main__":
    # Initialize the agent (uses config.ini from root by default)
    agent = AccuracyAgent()

    # Example 1: True fact - should PASS
    text1 = "The Eiffel Tower is located in Paris, France."
    result = agent.evaluate(text1)

    print(f"Status: {result['status']}")
    print(f"Score: {result['score']}")
    print(f"Reason: {result['reason']}\n")

    # Example 2: Common misconception - should FAIL
    text2 = "The Great Wall of China is visible from space with the naked eye."
    result2 = agent.evaluate(text2)

    print(f"Status: {result2['status']}")
    print(f"Score: {result2['score']}")
    print(f"Reason: {result2['reason']}\n")


