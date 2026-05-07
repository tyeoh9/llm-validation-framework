# LLM Validation Framework

A modular framework to validate LLM outputs across accuracy, relevancy, toxicity, and privacy.

## Install

Requires Python 3.11+

```bash
pip install -e .
```

**Optional extras:**
- `pip install -e ".[demo]"` — FastAPI demo server
- `pip install -e ".[test]"` — pytest + datasets
- `pip install -e ".[dev]"` — both

## Config

API keys are read from the environment or a `config.ini` file at the repo root (gitignored — never commit it).

**Option A — environment variable (recommended):**
```bash
export ANTHROPIC_API_KEY=your-key
```

**Option B — config.ini:**
```ini
[ANTHROPIC]
API_KEY=your-anthropic-api-key
```

Supported providers follow [litellm's naming](https://docs.litellm.ai/docs/providers).

## Agents

| Agent | What it does |
|---|---|
| `ToxicityAgent` | Three-layer harmful content check (profanity → toxicity model → semantic similarity). No API calls. |
| `PrivacyAgent` | Regex scan for SSN, credit cards, API keys, and optional system prompt leakage. No API calls. |
| `AccuracyAgent` | LLM-as-a-judge factual accuracy + relevancy, with optional RAG grounding. Requires API key. |
| `RelevancyAgent` | LLM-as-a-judge check that the answer addresses the question. Requires API key. |
| `BiasAgent` | LLM-as-a-judge scan for stereotypes and discriminatory language. Requires API key. |

`ToxicityAgent` and `PrivacyAgent` run fully locally with no external calls. The other three invoke an LLM on each evaluation.

**Notable constructor options:**
- `PrivacyAgent(system_prompt="...")` — also detects when the response leaks content from your system prompt
- `AccuracyAgent(rag=RAGProvider(retriever))` — grounds factual checks against your own corpus (see [RAG-Augmented Accuracy](#rag-augmented-accuracy))

## Usage

```python
from llm_validation_framework import ValidationFramework, LLMProvider, Pipe
from llm_validation_framework import ToxicityAgent, PrivacyAgent, AccuracyAgent

llm = LLMProvider(provider="anthropic", model="claude-haiku-4-5-20251001", key=api_key)
input_guardrail = Pipe(steps=[ToxicityAgent()], verbose=False)
output_guardrail = Pipe(steps=[ToxicityAgent(), PrivacyAgent(), AccuracyAgent()], verbose=False)

vf = ValidationFramework(llm=llm, input_guardrail=input_guardrail, output_guardrail=output_guardrail)
result = vf.validate("What is the Pacific Ocean?")
print(result["status"], result["score"])
```

See `examples/` for more usage patterns.

## Return value

`validate()` returns a nested dict:

```python
{
    "status": "PASS" | "FAIL",       # overall result
    "score": float,                   # average of input + output scores
    "input": {
        "status": "PASS" | "FAIL",
        "score": float,
        "results": [{"status": ..., "score": ..., "reason": ...}]
    },
    "output": {
        "status": "PASS" | "FAIL",
        "score": float,
        "results": [{"status": ..., "score": ..., "reason": ...}]
    }
}
```

Individual step results have `status` of `"PASS"`, `"FAIL"`, or `"TIMEOUT"`.

## RAG-Augmented Accuracy

Pass a retriever to `AccuracyAgent` to ground factual checks against your own corpus:

```python
from llm_validation_framework import AccuracyAgent, RAGProvider

retriever = your_vectorstore.as_retriever()  # any object with .invoke(query) -> List[Document]
accuracy = AccuracyAgent(rag=RAGProvider(retriever))
```

See `examples/accuracy_example.py` for a runnable version.

## Run the demo

The demo is a FastAPI backend + static web UI.

**Terminal 1 — API server:**
```bash
uvicorn demo.api_server:app --host 127.0.0.1 --port 5050
```

**Terminal 2 — UI:**
```bash
python demo/serve_ui.py
```

Open `http://127.0.0.1:8000` in your browser.

## Contributors
- Hitha Shri Nagaruru
- James Wu
- Lewis Lui
- Thomas Yeoh

## License

MIT — see [LICENSE](LICENSE)
