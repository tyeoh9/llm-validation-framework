# api_server.py — sketch; adjust imports/paths
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from components.pipe.Pipe import Pipe
from components.toxicityagent.ToxicityAgent import ToxicityAgent
from components.accuracy.AccuracyAgent import AccuracyAgent
from components.relevancy.RelevancyAgent import RelevancyAgent

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
        _pipe = Pipe(steps=[ToxicityAgent(), AccuracyAgent(config_path=None), RelevancyAgent(config_path=None)])
    return _pipe

class ValidateBody(BaseModel):
    question: str
    answer: str

@app.get("/validate/stream")
async def validate_stream(question: str, answer: str):
    async def event_generator():
        pipe = get_pipe()
        scores = []
        loop = asyncio.get_event_loop()

        for idx, step in enumerate(pipe.steps, start=1):
            name = getattr(step, "name", step.__class__.__name__)
            yield f"data: {json.dumps({'type': 'step_start', 'step': idx, 'name': name})}\n\n"

            queue: asyncio.Queue = asyncio.Queue()

            def on_progress(message, _q=queue):
                loop.call_soon_threadsafe(_q.put_nowait, message)

            task = asyncio.ensure_future(
                asyncio.to_thread(step.evaluate, {"question": question, "answer": answer}, on_progress=on_progress)
            )

            while not task.done():
                await asyncio.sleep(0.05)
                while not queue.empty():
                    msg = queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'step_phase', 'step': idx, 'message': msg})}\n\n"

            while not queue.empty():
                msg = queue.get_nowait()
                yield f"data: {json.dumps({'type': 'step_phase', 'step': idx, 'message': msg})}\n\n"

            result = await task
            scores.append(float(result.get("score", 0.0)))
            yield f"data: {json.dumps({'type': 'step_done', 'step': idx, 'name': name, 'status': result.get('status', ''), 'score': float(result.get('score', 0.0)), 'reason': result.get('reason') or ''})}\n\n"
            if result.get("status") == "FAIL":
                break

        overall = sum(scores) / len(scores) if scores else 0.0
        yield f"data: {json.dumps({'type': 'done', 'overall_score': overall})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/validate")
def validate(body: ValidateBody):
    pipe = get_pipe()
    results = pipe.evaluate({"question": body.question, "answer": body.answer})
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