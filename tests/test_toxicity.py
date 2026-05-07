"""Unit tests for ToxicityAgent layers."""

import pytest

from llm_validation_framework import ToxicityAgent


@pytest.fixture(scope="module")
def agent():
    return ToxicityAgent()


def test_deterministic_layer_clean(agent):
    result = agent.deterministic_layer("Hello, how are you?")
    assert result["status"] == "PASS"
    assert result["score"] == 0.0


def test_deterministic_layer_profanity(agent):
    result = agent.deterministic_layer("This is bullshit")
    assert result["status"] == "FAIL"
    assert result["score"] == 1.0


def test_probabilistic_layer_clean(agent):
    result = agent.probabilistic_layer("Hello, how are you?")
    assert result["status"] == "PASS"
    assert result["score"] < 0.5


def test_probabilistic_layer_toxic(agent):
    result = agent.probabilistic_layer("I hate you and want you dead")
    assert result["score"] >= 0.0


def test_semantic_layer_clean(agent):
    result = agent.semantic_layer("The weather is nice today")
    assert result["status"] == "PASS"


def test_evaluate_clean(agent):
    result = agent.evaluate("The Pacific Ocean is the largest ocean on Earth.")
    assert result["status"] == "PASS"
    assert 0.0 <= result["score"] <= 1.0


def test_evaluate_returns_dict(agent):
    result = agent.evaluate({"answer": "Hello world"})
    assert "status" in result
    assert "score" in result
    assert result["status"] in {"PASS", "FAIL"}
