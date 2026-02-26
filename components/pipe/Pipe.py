class Pipe:
    """Sequentially evaluates a list of steps, stopping early on the first failure."""

    def __init__(self, steps: list):
        self.steps = steps

    def evaluate(self, text: str) -> list[dict]:
        # Assumes all steps return {"status": "success"/"fail", "score": float}
        results = []
        for step in self.steps:
            result = step.evaluate(text)
            results.append(result)
            if result["status"] == "fail":
                return results
        return results