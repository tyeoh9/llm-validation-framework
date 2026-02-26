import sys
from pathlib import Path

from deepeval.models import GeminiModel, AnthropicModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Add root directory to path to import config_loader
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config_loader import load_api_key


class AccuracyAgent:
    """Uses LLM-as-a-judge to check factual accuracy of a statement."""

    def __init__(self, config_path: str | None = None):
        api_key = load_api_key(config_path)
        model = AnthropicModel(model="claude-3-5-haiku-latest", api_key=api_key)

        self.equivalence_metric = GEval(
            name="Text Equivalence",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=model,
            threshold=0.5,
        )

    def evaluate(self, text1: str, text2: str) -> dict:
        """Evaluate the equivalence between two text strings and return a score and reason."""

        test_case = LLMTestCase(
            input="Determine if the actual output is semantically equivalent to the expected output.",
            actual_output=text1,
            expected_output=text2,
        )
        self.equivalence_metric.measure(test_case)
        return {
            "score": self.equivalence_metric.score,
            "reason": self.equivalence_metric.reason,
        }