
# from ragprovider import RAGProvider
from llmprovider import LLMProvider
from pipe import Pipe

# TODO: Integrate RAG component once RAGProvider class has been implemented

class ValidationFramework:
    """Runs the entire validation pipeline from user query to final output."""

    def __init__(self, llm: LLMProvider, input_guardrail: Pipe, output_guardrail: Pipe):
        self.llm = llm
        # self.rag = rag
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail

    '''
    example of 'results' from Pipe class
    [
        {'status': 'PASS', 'score': 0.9761367253959179},
        {'status': 'FAIL',
        'score': 0.0,
        'reason': 'The actual output factually contradicts the expected output: it '
                    "claims Shanghai is China's capital when the evidence text only "
                    "describes Shanghai's financial and economic significance without "
                    "mentioning it as a capital, and Beijing is actually China's "
                    'capital.'}
    ]
    '''

    def validate(self, query: str):
        # TODO: Return metrics/scores rather than just PASS/FAIL message
        
        # 1. Validate input
        input_guard_result = self._check_failure(query)
        if input_guard_result == "FAIL":
            return "Validation failed at input guardrail"

        # 2. Call LLM
        response = self.llm.call_api(query)

        # 3. Validate output
        output_guard_result = self._check_failure(response)
        if output_guard_result == "FAIL":
            return "Validation failed at output guardrail"

        return "Validation passed!"


    def _check_failure(self, results: list[dict]) -> str:
        """Checks for failures/timeouts in this guardrail output"""
        for r in results:
            if r["status"] in ("FAIL", "TIMEOUT"):
                return "FAIL"
        return "PASS"