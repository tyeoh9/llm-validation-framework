# LLM Validation Framework

A general framework to validate LLMs.

## Install

**Requires Python 3.11+**

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

### Contributors:
- Hitha Shri Nagaruru
- James Wu
- Lewis Lui
- Thomas Yeoh