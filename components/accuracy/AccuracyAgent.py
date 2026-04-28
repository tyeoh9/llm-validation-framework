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

    def __init__(self, config_path: str | None = None, max_results: int = 10, provider: str = "anthropic", model: str = "claude-haiku-4-5-20251001"):
        self.config_path = config_path
        self._online = OnlineData(max_results=max_results)
        self._relevancy = RelevancyAgent(config_path=config_path, provider=provider, model=model)

        api_key = load_api_key(config_path, provider=provider.upper())
        llm_provider = LLMProvider(provider=provider, model=model, key=api_key)
        model = DeepEvalLLMProvider(llm_provider)

        self.equivalence_metric = GEval(
            name="Text Equivalence",
            evaluation_steps=[
                "First, determine whether the 'expected output' (evidence text) is topically relevant to the 'actual output' and the question.",
                "If the evidence is clearly off-topic or about a different subject, assign a score of 0.5 (neutral — evidence is not useful).",
                "If the evidence IS relevant, check ONLY whether the 'actual output' makes a specific factual claim that directly contradicts a fact in the evidence.",
                "A brief or single-word answer that identifies the correct entity (person, place, title, etc.) should score 0.8 or higher if it does not contradict the evidence.",
                "Absence of detail is NOT a contradiction — do not penalize an answer for being short or incomplete.",
                "Score below 0.5 only when the actual output asserts something that is factually wrong according to the evidence.",
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
            input="Does the actual output contradict any facts in the evidence text, or is the evidence irrelevant?",
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