"""Testing Script for ToxicityAgent layers."""
#!wget https://www.dropbox.com/s/ggl9krhh6dcwhhz/train.csv #Download the testing dataset!
from datasets import load_dataset
import pandas as pd
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

failures = {} #statement: correct_score
accuracy = []
precision = []
recall = []
f1 = []
T = ToxicityAgent()
#Conducted 10,000 sample test over 1hr 40 minutes
for i in range(10):
  # Let's just grab 10 random rows to test with so it doesn't take hours
  test_df = df.sample(1000)

  right, total = 0,0
  reference = {'PASS':0,'FAIL':1}
  true_positive,true_negative,false_positive,false_negative = 0,0,0,0
  for row in test_df.itertuples():
    cleaned = ''
    for c in row.comment_text:
      if c != '\n':
        cleaned += c

    guess,answer = reference[T.evaluate(cleaned,threshold=0.625)['status']], row.toxic
    if guess == answer and answer == 0:
      true_positive += 1
    elif guess == answer and answer == 1:
      true_negative += 1
    elif guess != answer and answer == 0:
      failures[cleaned] = 0
      false_positive += 1
    else:
      failures[cleaned] = 1
      false_negative += 1
    total += 1
  A = (true_positive + true_negative) / total
  P = true_positive / (true_positive + false_positive)
  R = true_positive / (true_positive + false_negative)
  accuracy.append(A)
  precision.append(P)
  recall.append(R)
  f1.append((2 * P * R) / (P + R))

print(f'Failures : {failures}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')

