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
from components.onlinedata.OnlineData import OnlineData


class AccuracyAgent:
    """Uses LLM-as-a-judge to check factual accuracy of a statement."""

    def __init__(self, config_path: str | None = None, max_results: int = 10):
        self.config_path = config_path
        self._online = OnlineData(max_results=max_results)

        api_key = load_api_key(config_path)
        # TODO: Make this model agnostic (make LLMProvider class compatible with deepeval)
        model = AnthropicModel(model="claude-haiku-4-5", api_key=api_key)

        self.equivalence_metric = GEval(
            name="Text Equivalence",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=model,
            threshold=0.5,
        )

    def find_evidence(self, query: str) -> str:
        """Retrieve external evidence for a query (currently via OnlineData only)."""
        body, href = self._online.search(query)
        return f"[Source: {href}]\n{body}"

    def evaluate(self, text: str) -> dict:
        """Evaluate a single text against external evidence."""
        evidence = self.find_evidence(text)

        test_case = LLMTestCase(
            input="Determine if the actual output is semantically consistent with the evidence text.",
            actual_output=text,
            expected_output=evidence,
        )
        self.equivalence_metric.measure(test_case)

        score = float(self.equivalence_metric.score or 0.0)
        threshold = float(getattr(self.equivalence_metric, "threshold", 0.5))
        status = "PASS" if score >= threshold else "FAIL"
        reason = getattr(self.equivalence_metric, "reason", "")

        return {"status": status, "score": score, "reason": reason}