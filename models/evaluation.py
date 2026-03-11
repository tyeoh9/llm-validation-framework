from typing import Literal, NotRequired, TypedDict


class EvaluationResult(TypedDict):
    status: Literal["PASS", "FAIL"]
    score: float
    reason: NotRequired[str]


class GuardrailSummary(TypedDict):
    status: Literal["PASS", "FAIL"]
    score: float
    reason: NotRequired[str]
    results: NotRequired[list[EvaluationResult]]


class ValidationSummary(TypedDict):
    input: GuardrailSummary
    output: GuardrailSummary
    status: Literal["PASS", "FAIL"]
    score: float
