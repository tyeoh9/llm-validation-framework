class Pipe:
    """Sequentially evaluates a list of steps, stopping early on the first failure."""

    def __init__(self, steps: list):
        self.steps = steps

    def evaluate(self, payload: dict) -> list[dict]:
        results = []
        for step in self.steps:
            result = step.evaluate(payload)
            results.append(result)
            if result.get("status") == "fail":
                return results
        return results