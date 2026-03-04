from typing import Literal, NotRequired, TypedDict


class EvaluationResult(TypedDict):
    status: Literal["PASS", "FAIL"]
    score: float
    reason: NotRequired[str]

