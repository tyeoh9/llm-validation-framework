
# from ragprovider import RAGProvider
from llmprovider import LLMProvider
from pipe import Pipe
from models import EvaluationResult, GuardrailSummary, ValidationSummary

# TODO: Integrate RAG component once RAGProvider class has been implemented

class ValidationFramework:
    """Runs the entire validation pipeline from user query to final output."""

    def __init__(self, llm: LLMProvider, input_guardrail: Pipe, output_guardrail: Pipe):
        self.llm = llm
        # self.rag = rag
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail

    def validate(self, query: str) -> ValidationSummary:
        """Run input guardrail -> LLM -> output guardrail and return structured results."""
        input_results = self.input_guardrail.evaluate(query)
        input_summary = self._summarize_results(input_results)

        if input_summary["status"] == "FAIL":
            output_summary: GuardrailSummary = {
                "status": "FAIL",
                "score": 0.0,
                "reason": "Skipped: input guardrail failed",
                "results": [],
            }
            final_score = (input_summary["score"] + output_summary["score"]) / 2.0
            return {
                "input": input_summary,
                "output": output_summary,
                "status": "FAIL",
                "score": final_score,
            }

        response = self.llm.call_api(query)

        output_results = self.output_guardrail.evaluate(response)
        output_summary = self._summarize_results(output_results)

        final_status = (
            "PASS"
            if input_summary["status"] == "PASS" and output_summary["status"] == "PASS"
            else "FAIL"
        )
        final_score = (input_summary["score"] + output_summary["score"]) / 2.0

        return {
            "input": input_summary,
            "output": output_summary,
            "status": final_status,
            "score": final_score,
        }

    def _check_failure(self, results: list[EvaluationResult]) -> str:
        """Checks for failures/timeouts in this guardrail output."""
        for r in results:
            if r["status"] in ("FAIL", "TIMEOUT"):
                return "FAIL"
        return "PASS"

    def _average_score(self, results: list[EvaluationResult]) -> float:
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            total += float(r.get("score", 0.0))
        return total / len(results)

    def _summarize_results(self, results: list[EvaluationResult]) -> GuardrailSummary:
        status = "FAIL" if self._check_failure(results) == "FAIL" else "PASS"
        score = self._average_score(results)
        return {"status": status, "score": score, "results": results}
