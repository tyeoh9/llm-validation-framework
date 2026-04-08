# api_server.py — sketch; adjust imports/paths
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from components.pipe.Pipe import Pipe
from components.toxicityagent.ToxicityAgent import ToxicityAgent
from components.accuracy.AccuracyAgent import AccuracyAgent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later, e.g. ["http://127.0.0.1:8000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipe = None

def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = Pipe(steps=[ToxicityAgent(), AccuracyAgent(config_path=None)])
    return _pipe

class ValidateBody(BaseModel):
    question: str
    answer: str

@app.post("/validate")
def validate(body: ValidateBody):
    pipe = get_pipe()
    results = pipe.evaluate(body.answer)
    out = []
    for step, r in zip(pipe.steps, results):
        out.append({
            "name": getattr(step, "name", step.__class__.__name__),
            "status": r.get("status", "?"),
            "score": float(r.get("score", 0.0)),
            "reason": r.get("reason") or "",
        })
    overall = sum(s["score"] for s in out) / len(out) if out else 0.0
    return {"overall_score": overall, "steps": out}