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
from components.relevancy.RelevancyAgent import RelevancyAgent
from models import EvaluationResult

RELEVANCY_WEIGHT = 0.4
FACTUAL_WEIGHT = 0.6


class AccuracyAgent:
    """Checks both relevancy and factual accuracy of an LLM answer."""

    name = "Accuracy check"

    def __init__(self, config_path: str | None = None, provider: str = "anthropic", model: str = "claude-haiku-4-5-20251001"):
        self.config_path = config_path
        self._relevancy = RelevancyAgent(config_path=config_path, provider=provider, model=model)

        api_key = load_api_key(config_path, provider=provider.upper())
        llm_provider = LLMProvider(provider=provider, model=model, key=api_key)
        model = DeepEvalLLMProvider(llm_provider)

        self.factual_metric = GEval(
            name="Factual Accuracy",
            evaluation_steps=[
                "Using your own knowledge, assess whether the actual output is a factually correct answer to the input question.",
                "A brief or single-word answer that correctly identifies the right entity, person, place, or title should be treated as fully correct.",
                "If you are uncertain whether the answer is correct, lean toward a higher score rather than penalizing by default.",
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
        """Run relevancy + factual checks and return a combined result."""
        question = data["question"] if isinstance(data, dict) else ""
        answer = data["answer"] if isinstance(data, dict) else data

        # --- Relevancy ---
        if on_progress:
            on_progress("Checking answer relevancy...")
        rel_result = self._relevancy.evaluate(data)
        rel_score = float(rel_result.get("score", 0.0))
        rel_reason = rel_result.get("reason", "")

        # --- Factual accuracy ---
        if on_progress:
            on_progress("Consulting judge model...")

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
        )
        self.factual_metric.measure(test_case)

        fact_score = float(self.factual_metric.score or 0.0)
        fact_reason = getattr(self.factual_metric, "reason", "")

        combined = RELEVANCY_WEIGHT * rel_score + FACTUAL_WEIGHT * fact_score
        status = "PASS" if combined >= 0.5 else "FAIL"

        return {
            "status": status,
            "score": combined,
            "reason": (
                f"Relevancy ({rel_score:.2f}): {rel_reason} | "
                f"Factual ({fact_score:.2f}): {fact_reason}"
            ),
        }