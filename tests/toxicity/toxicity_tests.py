from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from better_profanity import profanity
from detoxify import Detoxify
import torch

import sys
import logging
from pathlib import Path

from detoxify import Detoxify
from better_profanity import profanity
from sentence_transformers import SentenceTransformer, util
import torch


class ToxicityAgent:
    """Toxicity and policy-safety checks with deterministic, model, and semantic layers."""

    name = "Harmful content check"

    def __init__(self):
        self._detoxify_model = None
        self._sentence_model = None

    def deterministic_layer(
        self, statement: str, custom_bad_words: list[str] | None = None
    ):
        # Use the library's default profanity list, optionally extended with custom words
        if custom_bad_words:
            #default_words = set(profanity.get_profane_words())
            #default_words.update(custom_bad_words)
            profanity.add_censor_words(custom_bad_words)

        flagged = profanity.contains_profanity(statement)
        status = "FAIL" if flagged else "PASS"
        score = 1.0 if flagged else 0.0
        return {"status": status, "score": score}

    def probabilistic_layer(self, statement: str, threshold: float = 0.5):
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
        threshold: float = 0.1,
    ):
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

    def evaluate(self, statement: str, threshold: float = 0.5, on_progress=None):
        if on_progress: on_progress("Scanning for explicit language...")
        det_result = self.deterministic_layer(statement)
        if on_progress: on_progress("Running toxicity model...")
        prob_result = self.probabilistic_layer(statement, threshold=threshold)
        if on_progress: on_progress("Checking semantic similarity...")
        sem_result = self.semantic_layer(statement, threshold=threshold)

        # 1.0 means no risk, 0.0 means high risk
        risk_score = 1.0 - (
            0.2 * det_result["score"] +
            0.4 * prob_result["score"] +
            0.4 * sem_result["score"]
        )

        status = "FAIL" if risk_score < threshold else "PASS"
        return {"status": status, "score": float(risk_score)}

if __name__ == "__main__":
    # Convert the "train" split of the dataset into a Pandas DataFrame
    df = pd.read_csv("train.csv")

    # Let's just grab 10 random rows to test with so it doesn't take hours
    test_df = df.sample(100)
    T = ToxicityAgent()
    right, total = 0,0
    reference = {'PASS':0,'FAIL':1}
    true_positive,true_negative,false_positive,false_negative = 0,0,0,0
    for row in test_df.itertuples():
        cleaned = ''
        for c in row.comment_text:
            if c != '\n':
            cleaned += c

        guess,answer = reference[T.evaluate(cleaned)['status']], row.toxic
        if guess == answer and answer == 0:
            true_positive += 1
        elif guess == answer and answer == 1:
            true_negative += 1
        elif guess != answer and answer == 0:
            false_positive += 1
        else:
            false_negative += 1 
        total += 1
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')