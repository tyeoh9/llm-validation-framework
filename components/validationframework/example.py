"""Example usage of ValidationFramework (input + output guardrails).

RAG-augmented accuracy (optional):
    If you have a document corpus, pass a RAGProvider to AccuracyAgent so the
    judge uses retrieved context as ground truth instead of its own knowledge.

        from components.rag.RAG import RAG
        from components.rag.RAGProvider import RAGProvider

        rag_instance = RAG(data_dir="path/to/pdfs")
        rag_instance.build_or_load_vectorstore()
        retriever = rag_instance._vectorstore.as_retriever()
        rag = RAGProvider(retriever)

        output_guardrail = Pipe(steps=[ToxicityAgent(), AccuracyAgent(rag=rag)], verbose=False)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_api_key
from components.llmprovider.LLMProvider import LLMProvider
from components.pipe.Pipe import Pipe
from components.toxicityagent.ToxicityAgent import ToxicityAgent
from components.accuracy.AccuracyAgent import AccuracyAgent
from components.validationframework.ValidationFramework import ValidationFramework


def agent_names(pipe: Pipe) -> str:
    return ", ".join(step.name for step in pipe.steps)


def main():
    api_key = load_api_key(provider="ANTHROPIC")
    llm = LLMProvider(provider="anthropic", model="claude-3-haiku-20240307", key=api_key)

    input_guardrail = Pipe(steps=[ToxicityAgent()], verbose=False)
    output_guardrail = Pipe(steps=[ToxicityAgent(), AccuracyAgent()], verbose=False)

    vf = ValidationFramework(
        llm=llm,
        input_guardrail=input_guardrail,
        output_guardrail=output_guardrail,
    )

    query = "What is the pacific ocean?"
    result = vf.validate(query)

    print("Validation result")
    print(f"Input Guardrail  [{agent_names(input_guardrail)}]:  status={result['input']['status']} score={result['input']['score']:.2f}")
    print(f"Output Guardrail [{agent_names(output_guardrail)}]: status={result['output']['status']} score={result['output']['score']:.2f}")
    print(f"Final: status={result['status']} score={result['score']:.2f}")


if __name__ == "__main__":
    main()
