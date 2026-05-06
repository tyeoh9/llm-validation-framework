# LLM Validation Framework

A modular framework to validate LLM outputs across accuracy, relevancy, toxicity, and privacy.

## Install

Requires Python 3.11+

```bash
pip install -e .
```

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

## Run the demo

The demo is a FastAPI backend + static web UI. Install demo dependencies first:

```bash
pip install -e ".[demo]"
```

**Terminal 1 — API server:**
```bash
uvicorn demo.api_server:app --host 127.0.0.1 --port 5050
```

**Terminal 2 — UI:**
```bash
python demo/serve_ui.py
```

Open `http://127.0.0.1:8000` in your browser.

## Repository structure

```
llm_validation_framework/   installable Python package
├── validation_framework.py     main interface (ValidationFramework)
├── pipe.py                     sequential pipeline runner (Pipe)
├── llm_provider.py             LLM abstraction (LLMProvider)
├── toxicity_agent.py           3-layer toxicity check
├── privacy_agent.py            PII / secret detection
├── accuracy_agent.py           relevancy + factual accuracy
├── relevancy_agent.py          LLM-as-judge relevancy
├── bias_agent.py               LLM-as-judge bias detection
├── online_data.py              DuckDuckGo search + BM25 ranking
├── config_loader.py            API key loader
└── rag_provider.py             RAGProvider adapter (plug in any retriever)

demo/                       web demo (not part of the package)
├── api_server.py               FastAPI backend
├── serve_ui.py                 static file server
└── ui/                         HTML/JS/CSS frontend

examples/                   standalone usage scripts
tests/                      pytest test suite
```

## Contributors
- Hitha Shri Nagaruru
- James Wu
- Lewis Lui
- Thomas Yeoh
