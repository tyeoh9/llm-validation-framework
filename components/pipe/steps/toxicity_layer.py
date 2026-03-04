"""Toxicity step: runs ToxicityAgent on the LLM output. Uses deterministic + semantic layer."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.toxicityagent.ToxicityAgent import ToxicityAgent


class ToxicityStep:
    """Step that runs ToxicityAgent on the given text (LLM output). Returns pipeline-style result."""

    def __init__(self, illegal_categories: list[str] | None = None, semantic_threshold: float = 0.5):
        self.agent = ToxicityAgent()
        self.illegal_categories = illegal_categories or []
        self.semantic_threshold = semantic_threshold

    def evaluate(self, payload: dict) -> dict:
        # payload = {"question": str, "answer": str}
        answer = payload.get("answer", "")
        if not answer:
            return {"status": "fail", "score": 0.0, "reason": "Empty LLM output"}

        # Deterministic: profanity check
        det_result = self.agent.deterministic_layer(answer)
        if det_result != "Okay Statement":
            return {"status": "fail", "score": 0.0, "reason": f"Toxicity (deterministic): {det_result}"}

        # Optional: semantic policy check
        if self.illegal_categories:
            sem_result, sem_score = self.agent.semantic_layer(
                answer, self.illegal_categories, threshold=self.semantic_threshold
            )
            if sem_result == "FAIL":
                return {"status": "fail", "score": float(sem_score), "reason": f"Semantic policy violation (score={sem_score:.3f})"}

        return {"status": "success", "score": 1.0, "reason": "Toxicity checks passed"}