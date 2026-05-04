import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_api_key
from components.llmprovider.LLMProvider import LLMProvider, DeepEvalLLMProvider
from models import EvaluationResult


class RelevancyAgent:
    """Uses LLM-as-a-judge to check how relevant the LLM output is to the user's question."""

    name = "Relevancy check"

    def __init__(self, config_path: str | None = None, provider: str = "anthropic", model: str = "claude-haiku-4-5-20251001"):
        api_key = load_api_key(config_path, provider=provider.upper())
        llm_provider = LLMProvider(provider=provider, model=model, key=api_key)
        model = DeepEvalLLMProvider(llm_provider)

        self.relevancy_metric = GEval(
            name="Answer Relevancy",
            evaluation_steps=[
                "Check whether the actual output directly addresses the question asked in the input.",
                "Penalise answers that go off-topic or provide information unrelated to the question.",
                "An answer may include additional helpful context, but its core must be relevant to the input.",
                "The reasoning should sacrifice grammar for concision - one sentence only.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=model,
            threshold=0.5,
            verbose_mode=False,
        )

    def evaluate(self, data, on_progress=None) -> EvaluationResult:
        """Evaluate how relevant the answer is to the question."""
        question = data["question"] if isinstance(data, dict) else ""
        answer = data["answer"] if isinstance(data, dict) else data

        if on_progress:
            on_progress("Checking answer relevancy...")

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
        )
        self.relevancy_metric.measure(test_case)

        score = float(self.relevancy_metric.score or 0.0)
        threshold = float(getattr(self.relevancy_metric, "threshold", 0.5))
        status = "PASS" if score >= threshold else "FAIL"
        reason = getattr(self.relevancy_metric, "reason", "")

        return {"status": status, "score": score, "reason": reason}
