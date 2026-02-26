import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import load_api_key
from components.llmprovider.LLMProvider import LLMProvider


if __name__ == "__main__":
    api_key = load_api_key(provider="ANTHROPIC")
    provider_name = "anthropic"
    model_name = "claude-3-haiku-20240307"

    llm = LLMProvider(provider=provider_name, model=model_name, key=api_key)

    prompt = "In one short sentence, say hello from Anthropic."
    print(f"Sending prompt to {provider_name}/{model_name}...")
    response = llm.call_api(prompt)

    print(response)
