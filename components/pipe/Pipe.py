class Pipe:
    """Sequentially evaluates a list of steps.

    Each step is any object with an `evaluate` method. Whatever `data` is passed to
    `Pipe.evaluate(data)` is forwarded unchanged to each step's `evaluate`.
    All steps run regardless of pass/fail — the user decides what to do with the results.
    """

    def __init__(self, steps: list, verbose: bool = True):
        self.steps = steps
        self.verbose = verbose

    def evaluate(self, data) -> list[dict]:
        results = []
        for idx, step in enumerate(self.steps, start=1):
            result = step.evaluate(data)
            results.append(result)

            if self.verbose and result.get("reason"):
                print(f"Step {idx} reason: {result['reason']}")

        return results