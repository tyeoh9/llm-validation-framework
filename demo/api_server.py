import asyncio
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_validation_framework import (
    Pipe, ToxicityAgent, PrivacyAgent, AccuracyAgent, BiasAgent,
)
from demo.chatbot import Chatbot

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_input_guardrail = None
_output_guardrail = None
_chatbot = None


def get_guardrails():
    global _input_guardrail, _output_guardrail
    if _input_guardrail is None:
        _input_guardrail = Pipe(steps=[ToxicityAgent(), BiasAgent()])
    if _output_guardrail is None:
        _output_guardrail = Pipe(
            steps=[ToxicityAgent(), PrivacyAgent(), AccuracyAgent(), BiasAgent()]
        )
    return _input_guardrail, _output_guardrail


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        _chatbot = Chatbot()
    return _chatbot


class ValidateBody(BaseModel):
    question: str
    answer: str


@app.get("/validate/stream")
async def validate_stream(question: str, answer: str):
    async def event_generator():
        input_guardrail, output_guardrail = get_guardrails()
        loop = asyncio.get_event_loop()

        # (step, data) pairs: input steps see only the question,
        # output steps see both question and answer
        sections = [
            ("Input Guardrail", input_guardrail.steps, question),
            ("Output Guardrail", output_guardrail.steps, {"question": question, "answer": answer}),
        ]

        input_scores = []
        output_scores = []
        step_idx = 0

        for section_name, steps, data in sections:
            yield f"data: {json.dumps({'type': 'section_start', 'name': section_name})}\n\n"

            for step in steps:
                step_idx += 1
                name = getattr(step, "name", step.__class__.__name__)
                yield f"data: {json.dumps({'type': 'step_start', 'step': step_idx, 'name': name})}\n\n"

                queue: asyncio.Queue = asyncio.Queue()

                def on_progress(message, _q=queue):
                    loop.call_soon_threadsafe(_q.put_nowait, message)

                task = asyncio.ensure_future(
                    asyncio.to_thread(step.evaluate, data, on_progress=on_progress)
                )

                while not task.done():
                    await asyncio.sleep(0.05)
                    while not queue.empty():
                        msg = queue.get_nowait()
                        yield f"data: {json.dumps({'type': 'step_phase', 'step': step_idx, 'message': msg})}\n\n"

                while not queue.empty():
                    msg = queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'step_phase', 'step': step_idx, 'message': msg})}\n\n"

                result = await task
                score = float(result.get("score", 0.0))

                if section_name == "Input Guardrail":
                    input_scores.append(score)
                else:
                    output_scores.append(score)

                yield f"data: {json.dumps({'type': 'step_done', 'step': step_idx, 'name': name, 'status': result.get('status', ''), 'score': score, 'reason': result.get('reason') or ''})}\n\n"

        input_avg = sum(input_scores) / len(input_scores) if input_scores else 0.0
        output_avg = sum(output_scores) / len(output_scores) if output_scores else 0.0
        overall = (input_avg + output_avg) / 2.0
        yield f"data: {json.dumps({'type': 'done', 'overall_score': overall})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/validate")
def validate(body: ValidateBody):
    input_guardrail, output_guardrail = get_guardrails()

    input_results = input_guardrail.evaluate(body.question)
    output_results = output_guardrail.evaluate({"question": body.question, "answer": body.answer})

    out = []
    for step, r in zip(input_guardrail.steps, input_results):
        out.append({
            "guardrail": "input",
            "name": getattr(step, "name", step.__class__.__name__),
            "status": r.get("status", "?"),
            "score": float(r.get("score", 0.0)),
            "reason": r.get("reason") or "",
        })
    for step, r in zip(output_guardrail.steps, output_results):
        out.append({
            "guardrail": "output",
            "name": getattr(step, "name", step.__class__.__name__),
            "status": r.get("status", "?"),
            "score": float(r.get("score", 0.0)),
            "reason": r.get("reason") or "",
        })

    input_scores = [s["score"] for s in out if s["guardrail"] == "input"]
    output_scores = [s["score"] for s in out if s["guardrail"] == "output"]
    input_avg = sum(input_scores) / len(input_scores) if input_scores else 0.0
    output_avg = sum(output_scores) / len(output_scores) if output_scores else 0.0
    overall = (input_avg + output_avg) / 2.0

    return {"overall_score": overall, "steps": out}


@app.get("/chat/stream")
async def chat_stream(question: str):
    async def event_generator():
        bot = get_chatbot()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        sentinel = object()

        def _produce():
            for token in bot.stream(question):
                loop.call_soon_threadsafe(queue.put_nowait, token)
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        asyncio.ensure_future(asyncio.to_thread(_produce))

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
