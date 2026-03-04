from typing import Literal, TypedDict


class EvaluationResult(TypedDict):
    status: Literal["PASS", "FAIL"]
    score: float

