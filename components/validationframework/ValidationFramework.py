
# from ragprovider import RAGProvider
from llmprovider import LLMProvider
from pipe import Pipe

# TODO: Integrate RAG component once RAGProvider class has been implemented

class ValidationFramework:
    def __init__(self, llm: LLMProvider, input_guardrail: Pipe, output_guardrail: Pipe):
        self.llm = llm
        # self.rag = rag
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail

    