class Pipe:
    """Sequentially evaluates a list of steps, stopping early on the first failure.

    Each step is any object with an `evaluate` method. Whatever `data` is passed to
    `Pipe.evaluate(data)` is forwarded unchanged to each step's `evaluate`.
    """

    def __init__(self, steps: list):
        self.steps = steps

    def evaluate(self, data) -> list[dict]:
        results = []
        for step in self.steps:
            result = step.evaluate(data)
            results.append(result)
            if result.get("status") == "FAIL":
                return results
        return results