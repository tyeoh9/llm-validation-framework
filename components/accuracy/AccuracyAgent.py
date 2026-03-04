import sys
from pathlib import Path

from deepeval.models import GeminiModel, AnthropicModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Add repository root to path to import shared modules
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_api_key
from models import EvaluationResult


class AccuracyAgent:
    """Uses LLM-as-a-judge to check factual accuracy of a statement."""

    def __init__(self, config_path: str | None = None):
        api_key = load_api_key(config_path)
        # TODO: Make this model agnostic (make LLMProvider class compatible with deepeval)
        model = AnthropicModel(model="claude-haiku-4-5", api_key=api_key)

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

    def evaluate(self, text1: str, text2: str) -> EvaluationResult:
        """Evaluate the equivalence between two text strings and return PASS/FAIL + score."""

        test_case = LLMTestCase(
            input="Determine if the actual output is semantically equivalent to the expected output.",
            actual_output=text1,
            expected_output=text2,
        )
        self.equivalence_metric.measure(test_case)
        score = float(self.equivalence_metric.score or 0.0)
        threshold = float(getattr(self.equivalence_metric, "threshold", 0.5))
        status = "PASS" if score >= threshold else "FAIL"

        return {"status": status, "score": score}