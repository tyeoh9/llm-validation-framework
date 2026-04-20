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
from components.onlinedata.OnlineData import OnlineData
from components.llmprovider.LLMProvider import LLMProvider, DeepEvalLLMProvider
from components.relevancy.RelevancyAgent import RelevancyAgent
from models import EvaluationResult

RELEVANCY_WEIGHT = 0.4
FACTUAL_WEIGHT = 0.6


class AccuracyAgent:
    """Checks both relevancy and factual accuracy of an LLM answer."""

    name = "Accuracy check"

    def __init__(self, config_path: str | None = None, max_results: int = 10):
        self.config_path = config_path
        self._online = OnlineData(max_results=max_results)
        self._relevancy = RelevancyAgent(config_path=config_path)

        api_key = load_api_key(config_path)
        llm_provider = LLMProvider(provider="anthropic", model="claude-haiku-4-5-20251001", key=api_key)
        model = DeepEvalLLMProvider(llm_provider)

        self.equivalence_metric = GEval(
            name="Text Equivalence",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "Contradicting opinions are OK but contradict facts are not.",
                "The reasoning should sacrifice grammar for concision - one sentence only."
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=model,
            threshold=0.5,
            verbose_mode=False,
        )

    def find_evidence(self, query: str) -> str | None:
        """Retrieve external evidence for a query (currently via OnlineData only).
        Returns None if evidence retrieval fails."""
        body, href = self._online.search(query)
        if body is None:
            return None
        return f"[Source: {href}]\n{body}"

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
            on_progress("Fetching supporting evidence...")
        evidence = self.find_evidence(question or answer)

        if evidence is None:
            combined = RELEVANCY_WEIGHT * rel_score
            return {
                "status": "FAIL" if combined < 0.5 else "PASS",
                "score": combined,
                "reason": (
                    f"Relevancy ({rel_score:.2f}): {rel_reason} | "
                    f"Factual: skipped (evidence retrieval failed)."
                ),
            }

        if on_progress:
            on_progress("Consulting judge model...")

        test_case = LLMTestCase(
            input="Determine if the actual output is semantically consistent with the evidence text.",
            actual_output=answer,
            expected_output=evidence,
        )
        self.equivalence_metric.measure(test_case)

        fact_score = float(self.equivalence_metric.score or 0.0)
        fact_reason = getattr(self.equivalence_metric, "reason", "")

        combined = RELEVANCY_WEIGHT * rel_score + FACTUAL_WEIGHT * fact_score
        status = "PASS" if combined >= 0.5 else "FAIL"

        reason = (
            f"Relevancy ({rel_score:.2f}): {rel_reason} | "
            f"Factual ({fact_score:.2f}): {fact_reason}"
        )

        return {"status": status, "score": combined, "reason": reason}