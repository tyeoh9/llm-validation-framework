from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from llm_validation_framework.config_loader import load_api_key
from llm_validation_framework.llm_provider import LLMProvider, DeepEvalLLMProvider
from llm_validation_framework.relevancy_agent import RelevancyAgent
from llm_validation_framework.models import EvaluationResult

if TYPE_CHECKING:
    from llm_validation_framework.rag_provider import RAGProvider

RELEVANCY_WEIGHT = 0.4
FACTUAL_WEIGHT = 0.6


class AccuracyAgent:
    """Checks both relevancy and factual accuracy of an LLM answer."""

    name = "Accuracy check"

    def __init__(
        self,
        config_path: str | None = None,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        rag: Optional[RAGProvider] = None,
    ):
        self.config_path = config_path
        self.rag = rag
        self._relevancy = RelevancyAgent(config_path=config_path, provider=provider, model=model)

        api_key = load_api_key(config_path, provider=provider.upper())
        llm_provider = LLMProvider(provider=provider, model=model, key=api_key)
        judge_model = DeepEvalLLMProvider(llm_provider)

        if rag:
            evaluation_steps = [
                "Using the provided context as the source of truth, assess whether the actual output is a factually correct answer to the input question.",
                "Penalize answers that contradict or are unsupported by the context, even if they seem plausible from general knowledge.",
                "A brief or single-word answer that correctly matches the context should be treated as fully correct.",
                "The reasoning should sacrifice grammar for concision - one sentence only.",
            ]
            evaluation_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ]
        else:
            evaluation_steps = [
                "Using your own knowledge, assess whether the actual output is a factually correct answer to the input question.",
                "A brief or single-word answer that correctly identifies the right entity, person, place, or title should be treated as fully correct.",
                "If you are uncertain whether the answer is correct, lean toward a higher score rather than penalizing by default.",
                "The reasoning should sacrifice grammar for concision - one sentence only.",
            ]
            evaluation_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ]

        self.factual_metric = GEval(
            name="Factual Accuracy",
            evaluation_steps=evaluation_steps,
            evaluation_params=evaluation_params,
            model=judge_model,
            threshold=0.5,
            verbose_mode=False,
        )

    def evaluate(self, data, on_progress=None) -> EvaluationResult:
        """Run relevancy + factual checks and return a combined result."""
        question = data["question"] if isinstance(data, dict) else ""
        answer = data["answer"] if isinstance(data, dict) else data

        if on_progress:
            on_progress("Checking answer relevancy...")
        rel_result = self._relevancy.evaluate(data)
        rel_score = float(rel_result.get("score", 0.0))
        rel_reason = rel_result.get("reason", "")

        if on_progress:
            on_progress("Consulting judge model...")

        context = None
        if self.rag:
            if on_progress:
                on_progress("Retrieving RAG context...")
            retrieved = self.rag.extract_content(question)
            if retrieved:
                context = [retrieved]

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context,
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
