"""Example usage of ValidationFramework (input + output guardrails).

RAG-augmented accuracy (optional):
    If you have a document corpus, pass a RAGProvider to AccuracyAgent so the
    judge uses retrieved context as ground truth instead of its own knowledge.
    Bring your own retriever — any object with .invoke(query: str) -> List[Document].

        from llm_validation_framework import RAGProvider

        rag = RAGProvider(your_retriever)
        output_guardrail = Pipe(steps=[ToxicityAgent(), AccuracyAgent(rag=rag)], verbose=False)
"""

from llm_validation_framework import (
    ValidationFramework,
    LLMProvider,
    Pipe,
    ToxicityAgent,
    AccuracyAgent,
)
from llm_validation_framework.config_loader import load_api_key


def agent_names(pipe: Pipe) -> str:
    return ", ".join(step.name for step in pipe.steps)


def main():
    api_key = load_api_key(provider="ANTHROPIC")
    llm = LLMProvider(provider="anthropic", model="claude-haiku-4-5-20251001", key=api_key)

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
