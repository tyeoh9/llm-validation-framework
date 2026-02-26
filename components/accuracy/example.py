"""Example usage of AccuracyAgent for text equivalence evaluation."""

from AccuracyAgent import AccuracyAgent

if __name__ == "__main__":
    # Initialize the agent (uses config.ini from root by default)
    agent = AccuracyAgent()

    # Example 1: Evaluate two similar texts
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on the mat."
    result = agent.evaluate(text1, text2)

    print(f"Score: {result['score']}")
    print(f"Reason: {result['reason']}\n")

    # Example 2: Evaluate two different texts
    text3 = "The weather is sunny today."
    text4 = "I went to the store to buy groceries."
    result2 = agent.evaluate(text3, text4)

    print(f"Score: {result2['score']}")
    print(f"Reason: {result2['reason']}")

