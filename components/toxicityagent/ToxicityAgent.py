from detoxify import Detoxify
from better_profanity import profanity
from sentence_transformers import SentenceTransformer, util
import torch


class ToxicityAgent:
    """Toxicity and policy-safety checks with deterministic, model, and semantic layers."""

    def __init__(self):
        self._detoxify_model = None
        self._sentence_model = None

    def deterministic_layer(self, statement: str, custom_bad_words: list[str] | None = None) -> str:
        words = custom_bad_words or []
        profanity.load_censor_words(words)
        if profanity.contains_profanity(statement):
            return "Illegal/Toxic Content Detected"
        return "Okay Statement"

    def probabilistic_layer(self, statement: str) -> dict:
        if self._detoxify_model is None:
            self._detoxify_model = Detoxify("original")
        return self._detoxify_model.predict(statement)

    def semantic_layer(
        self,
        statement: str,
        illegal_categories: list[str],
        threshold: float = 0.5,
    ) -> tuple[str, float]:
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        deny_embeddings = self._sentence_model.encode(
            illegal_categories, convert_to_tensor=True
        )
        user_embedding = self._sentence_model.encode(statement, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, deny_embeddings)
        max_score = torch.max(cosine_scores).item()

        if max_score > threshold:
            return "FAIL", max_score
        return "PASS", max_score
