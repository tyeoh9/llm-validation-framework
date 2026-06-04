from typing import Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class EvaluationResult(TypedDict):
    status: Literal["PASS", "FAIL", "TIMEOUT"]
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
