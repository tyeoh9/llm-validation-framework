"""
Live integration tests for OnlineData.search() using TriviaQA (rc.web split).

Fires real queries at DuckDuckGo and measures hit rate — i.e. how often the
returned snippet contains a known correct answer.

Run with:
    pytest tests/test_onlinedata_live.py -m integration -s

The -s flag prints the per-query hit/miss breakdown.
"""

import pytest

from llm_validation_framework import OnlineData

SAMPLE_SIZE = 25
HIT_RATE_THRESHOLD = 0.4


def load_trivia_qa_samples(n: int):
    from datasets import load_dataset

    dataset = load_dataset("trivia_qa", "rc.web", split="validation", trust_remote_code=True)
    samples = []
    for row in dataset:
        question = row["question"]
        answer_value = row["answer"]["value"]
        aliases = row["answer"].get("aliases", [])
        all_answers = list({answer_value} | set(aliases))
        samples.append({"question": question, "answers": all_answers})
        if len(samples) >= n:
            break
    return samples


def snippet_contains_answer(body: str, answers: list[str]) -> bool:
    body_lower = body.lower()
    return any(ans.lower() in body_lower for ans in answers)


@pytest.fixture(scope="module")
def trivia_samples():
    return load_trivia_qa_samples(SAMPLE_SIZE)


@pytest.fixture(scope="module")
def online_data():
    return OnlineData(max_results=10)


@pytest.mark.integration
def test_search_returns_result(online_data, trivia_samples):
    """search() returns a non-None body and href for at least one sample."""
    sample = trivia_samples[0]
    body, href = online_data.search(sample["question"])
    assert body is not None, "search() returned None for the first sample"
    assert len(body) > 0, "search() returned an empty body"
    assert href is not None, "search() returned None href"


@pytest.mark.integration
def test_search_hit_rate(online_data, trivia_samples, log_run):
    """Top-1 live search result contains the answer for >= HIT_RATE_THRESHOLD of queries."""
    hits = 0
    query_log = []

    for sample in trivia_samples:
        body, href = online_data.search(sample["question"])
        if body is None:
            query_log.append({
                "question": sample["question"],
                "expected": sample["answers"],
                "hit": False,
                "source": None,
                "note": "DDGS returned None",
            })
            continue
        hit = snippet_contains_answer(body, sample["answers"])
        if hit:
            hits += 1
        query_log.append({
            "question": sample["question"],
            "expected": sample["answers"],
            "hit": hit,
            "source": href,
        })

    hit_rate = hits / SAMPLE_SIZE
    print(f"\nLive search hit rate: {hit_rate:.2%} ({hits}/{SAMPLE_SIZE})")
    misses = [q for q in query_log if not q["hit"]]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for q in misses[:5]:
            print(f"  Q: {q['question']!r}  expected: {q['expected']}")

    log_run(
        component="live_search",
        sample_size=SAMPLE_SIZE,
        hits=hits,
        queries=query_log,
    )

    assert hit_rate >= HIT_RATE_THRESHOLD, (
        f"Hit rate {hit_rate:.2%} is below threshold {HIT_RATE_THRESHOLD:.2%}"
    )


@pytest.mark.integration
def test_search_graceful_on_failure(online_data):
    """search() returns (None, None) when DDGS raises, rather than crashing."""
    from unittest.mock import patch

    with patch.object(online_data.searcher, "text", side_effect=Exception("rate limited")):
        body, href = online_data.search("What is the capital of France?")

    assert body is None
    assert href is None
