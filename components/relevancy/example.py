"""Example usage of RelevancyAgent."""

from RelevancyAgent import RelevancyAgent

if __name__ == "__main__":
    agent = RelevancyAgent()

    # Example 1: Relevant answer — should PASS
    result = agent.evaluate({
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
    })
    print(f"Status: {result['status']}")
    print(f"Score:  {result['score']}")
    print(f"Reason: {result['reason']}\n")

    # Example 2: Irrelevant answer — should FAIL
    result2 = agent.evaluate({
        "question": "What is the capital of France?",
        "answer": "Python is a popular programming language used for web development.",
    })
    print(f"Status: {result2['status']}")
    print(f"Score:  {result2['score']}")
    print(f"Reason: {result2['reason']}\n")
