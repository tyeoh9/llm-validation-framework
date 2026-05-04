"""
Shared fixtures for onlinedata tests.

Provides a `log_run` fixture that appends a timestamped JSON entry to
tests/onlinedata/logs/results.jsonl after each metric-tracking test.

Log format (one JSON object per line):
{
    "timestamp": "2026-04-14T12:00:00.000000",
    "test":      "test_bm25_top1_hit_rate",
    "component": "bm25_ranking" | "live_search",
    "sample_size": 50,
    "hits": 23,
    "hit_rate": 0.46,
    "queries": [
        {"question": "...", "expected": [...], "hit": true,  "source": "https://..."},
        ...
    ]
}
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "results.jsonl"


@pytest.fixture
def log_run(request):
    """Return a callable that writes one result entry to the log file.

    Usage inside a test:
        def test_something(log_run):
            ...
            log_run(
                component="bm25_ranking",
                sample_size=50,
                hits=23,
                queries=[{"question": ..., "expected": ..., "hit": ..., "source": ...}],
            )
    """

    def _write(component: str, sample_size: int, hits: int, queries: list[dict]):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "test": request.node.name,
            "component": component,
            "sample_size": sample_size,
            "hits": hits,
            "hit_rate": round(hits / sample_size, 4) if sample_size else 0.0,
            "queries": queries,
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    return _write
