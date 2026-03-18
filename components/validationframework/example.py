"""Example usage of ValidationFramework (input + output guardrails)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_api_key
from components.llmprovider.LLMProvider import LLMProvider
from components.pipe.Pipe import Pipe
from components.toxicityagent.ToxicityAgent import ToxicityAgent
from components.validationframework.ValidationFramework import ValidationFramework


def main():
    api_key = load_api_key(provider="ANTHROPIC")
    llm = LLMProvider(provider="anthropic", model="claude-3-haiku-20240307", key=api_key)

    input_guardrail = Pipe(steps=[ToxicityAgent()])
    output_guardrail = Pipe(steps=[ToxicityAgent()])

    vf = ValidationFramework(
        llm=llm,
        input_guardrail=input_guardrail,
        output_guardrail=output_guardrail,
    )

    query = "In one short sentence, explain what the Pacific Ocean is."
    result = vf.validate(query)

    print("Validation result")
    print(f"Input:  status={result['input']['status']} score={result['input']['score']:.2f}")
    print(f"Output: status={result['output']['status']} score={result['output']['score']:.2f}")
    print(f"Final:  status={result['status']} score={result['score']:.2f}")


if __name__ == "__main__":
    main()
