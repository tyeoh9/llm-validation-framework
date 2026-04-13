import sys
from pathlib import Path

import litellm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_api_key


class Chatbot:
    """Single-turn chatbot that streams responses token by token.

    Designed for easy upgrade to multi-turn: swap `question` (str) for a
    messages list when chat history support is needed.
    """

    def __init__(self, provider: str = "anthropic", model: str = "claude-haiku-4-5-20251001",
                 config_path: str | None = None):
        self.provider = provider
        self.model = model
        api_key = load_api_key(config_path, provider=provider.upper())
        self.model_string = f"{provider}/{model}"
        self.api_key = api_key

    def stream(self, question: str):
        """Yield response tokens one at a time."""
        messages = [{"role": "user", "content": question}]
        response = litellm.completion(
            model=self.model_string,
            messages=messages,
            api_key=self.api_key,
            stream=True,
        )
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def ask(self, question: str) -> str:
        """Non-streaming convenience method. Returns the full response."""
        return "".join(self.stream(question))
