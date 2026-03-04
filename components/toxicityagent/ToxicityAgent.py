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
        # Use the library's default profanity list, optionally extended with custom words
        if custom_bad_words:
            default_words = set(profanity.get_profane_words())
            default_words.update(custom_bad_words)
            profanity.load_censor_words(list(default_words))

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
        illegal_categories: list[str] | None = None,
        threshold: float = 0.5,
    ) -> tuple[str, float]:
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

        if max_score > threshold:
            return "FAIL", max_score
        return "PASS", max_score

    def evaluate(self, statement: str, threshold: float = 0.5):
        layer_one = self.deterministic_layer(statement)
        layer_two = self.probabilistic_layer(statement)
        layer_three = self.semantic_layer(statement, threshold=threshold)
        return {
            "layer_one": layer_one,
            "layer_two": layer_two,
            "layer_three": layer_three,
        }