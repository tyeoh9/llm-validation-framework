# LLM Validation Framework

A modular framework to validate LLM outputs across accuracy, relevancy, toxicity, and privacy.

## Config

Create a `config.ini` at the repo root (already gitignored — never commit it):

```ini
[ANTHROPIC]
API_KEY=your-anthropic-api-key
```

## Install

Requires Python 3.11+

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

**Terminal 1 — API server:**
```bash
uvicorn api_server:app --host 127.0.0.1 --port 5050
```

**Terminal 2 — UI:**
```bash
python3 serve_ui.py
```

## Contributors
- Hitha Shri Nagaruru
- James Wu
- Lewis Lui
- Thomas Yeoh
