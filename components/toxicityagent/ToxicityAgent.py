import sys
import logging
from pathlib import Path

from detoxify import Detoxify
from better_profanity import profanity
from sentence_transformers import SentenceTransformer, util
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.getLogger("transformers").setLevel(logging.ERROR)

from models import EvaluationResult


class ToxicityAgent:
    """Toxicity and policy-safety checks with deterministic, model, and semantic layers."""

    name = "ToxicityAgent"

    def __init__(self):
        self._detoxify_model = None
        self._sentence_model = None

    def deterministic_layer(
        self, statement: str, custom_bad_words: list[str] | None = None
    ) -> EvaluationResult:
        # Use the library's default profanity list, optionally extended with custom words
        if custom_bad_words:
            default_words = set(profanity.get_profane_words())
            default_words.update(custom_bad_words)
            profanity.load_censor_words(list(default_words))

        flagged = profanity.contains_profanity(statement)
        status = "FAIL" if flagged else "PASS"
        score = 1.0 if flagged else 0.0
        return {"status": status, "score": score}

    def probabilistic_layer(self, statement: str, threshold: float = 0.5) -> EvaluationResult:
        if self._detoxify_model is None:
            self._detoxify_model = Detoxify("original")
        tox_scores = self._detoxify_model.predict(statement)
        max_tox_score = max(tox_scores.values()) if tox_scores else 0.0
        status = "FAIL" if max_tox_score > threshold else "PASS"
        return {"status": status, "score": float(max_tox_score)}

    def semantic_layer(
        self,
        statement: str,
        illegal_categories: list[str] | None = None,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        default_illegal_categories = [
            "hate speech",
            "self-harm encouragement",
            "sexual content involving minors",
            "violent or graphic harm",
            "terrorism or extremism",
            "instructions for illegal activities",
        ]
        categories = illegal_categories or default_illegal_categories

        deny_embeddings = self._sentence_model.encode(
            categories, convert_to_tensor=True
        )
        user_embedding = self._sentence_model.encode(statement, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, deny_embeddings)
        max_score = torch.max(cosine_scores).item()

        status = "FAIL" if max_score > threshold else "PASS"
        return {"status": status, "score": float(max_score)}

    def evaluate(self, statement: str, threshold: float = 0.5) -> EvaluationResult:
        det_result = self.deterministic_layer(statement)
        prob_result = self.probabilistic_layer(statement, threshold=threshold)
        sem_result = self.semantic_layer(statement, threshold=threshold)

        # 1.0 means no risk, 0.0 means high risk
        risk_score = 1.0 - max(
            float(det_result["score"]),
            float(prob_result["score"]),
            float(sem_result["score"]),
        )

        fail = any(r["status"] == "FAIL" for r in (det_result, prob_result, sem_result))
        status = "FAIL" if fail else "PASS"

        return {"status": status, "score": float(risk_score)}