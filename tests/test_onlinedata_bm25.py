"""
Offline unit tests for OnlineData.rank_results() using TriviaQA (rc.web split).

Uses pre-collected web snippets from the dataset — no network calls to DDGS.
Measures how often BM25 ranks a snippet containing the correct answer at the top.
"""

import pytest

from llm_validation_framework import OnlineData

SAMPLE_SIZE = 50
HIT_RATE_THRESHOLD = 0.4


def load_trivia_qa_samples():
    """Load a small sample from TriviaQA rc.web (train split)."""
    from datasets import load_dataset

    dataset = load_dataset("trivia_qa", "rc.web", split="train", trust_remote_code=True)
    samples = []
    for row in dataset:
        question = row["question"]
        answer_value = row["answer"]["value"]
        aliases = row["answer"].get("aliases", [])
        all_answers = list({answer_value} | set(aliases))

        sr = row.get("search_results", {})
        search_result_keys = list(sr.keys()) if isinstance(sr, dict) else []
        if not search_result_keys:
            continue

        num_results = len(sr[search_result_keys[0]])
        pages = []
        for i in range(num_results):
            body = sr.get("search_context", [""] * num_results)[i] or ""
            href = sr.get("url", [""] * num_results)[i] or ""
            if body:
                pages.append({"body": body, "href": href})

        if pages:
            samples.append({"question": question, "answers": all_answers, "pages": pages})

        if len(samples) >= SAMPLE_SIZE:
            break

    return samples


def snippet_contains_answer(body: str, answers: list[str]) -> bool:
    body_lower = body.lower()
    return any(ans.lower() in body_lower for ans in answers)


@pytest.fixture(scope="module")
def trivia_samples():
    return load_trivia_qa_samples()


@pytest.fixture(scope="module")
def online_data():
    return OnlineData()


def test_trivia_qa_loaded(trivia_samples):
    """Dataset loads and returns the expected number of samples."""
    assert len(trivia_samples) == SAMPLE_SIZE, (
        f"Expected {SAMPLE_SIZE} samples, got {len(trivia_samples)}"
    )


def test_bm25_top1_hit_rate(trivia_samples, online_data, log_run):
    """Top-1 BM25 result contains the answer for at least HIT_RATE_THRESHOLD of queries."""
    hits = 0
    query_log = []

    for sample in trivia_samples:
        ranked = online_data.rank_results(sample["question"], sample["pages"])
        top_page, _ = ranked[0]
        top_body = top_page["body"]
        hit = snippet_contains_answer(top_body, sample["answers"])
        if hit:
            hits += 1
        query_log.append({
            "question": sample["question"],
            "expected": sample["answers"],
            "hit": hit,
            "source": top_page.get("href", ""),
        })

    hit_rate = hits / len(trivia_samples)
    print(f"\nBM25 top-1 hit rate: {hit_rate:.2%} ({hits}/{len(trivia_samples)})")

    log_run(
        component="bm25_ranking",
        sample_size=len(trivia_samples),
        hits=hits,
        queries=query_log,
    )

    assert hit_rate >= HIT_RATE_THRESHOLD, (
        f"Hit rate {hit_rate:.2%} is below threshold {HIT_RATE_THRESHOLD:.2%}"
    )


def test_bm25_ranking_is_ordered(trivia_samples, online_data):
    """rank_results() always returns results in descending score order."""
    sample = trivia_samples[0]
    ranked = online_data.rank_results(sample["question"], sample["pages"])
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True), "Results are not sorted by score descending"


def test_bm25_returns_all_pages(trivia_samples, online_data):
    """rank_results() returns the same number of results as input pages."""
    sample = trivia_samples[0]
    ranked = online_data.rank_results(sample["question"], sample["pages"])
    assert len(ranked) == len(sample["pages"])
