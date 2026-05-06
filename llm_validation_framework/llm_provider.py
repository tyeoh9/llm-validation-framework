import litellm
from deepeval.models.base_model import DeepEvalBaseLLM


class LLMProvider:
    """
    A unified interface to call various LLM providers.
    To see which providers this class supports, visit https://docs.litellm.ai/docs/providers
    """

    def __init__(self, provider: str, model: str, key: str):
        self.model_string = f"{provider}/{model}"
        self.key = key

    def call_api(self, query: str) -> str:
        response = litellm.completion(
            model=self.model_string,
            messages=[{"role": "user", "content": query}],
            api_key=self.key
        )
        return response.choices[0].message.content


class DeepEvalLLMProvider(DeepEvalBaseLLM):
    """Adapter that makes LLMProvider compatible with deepeval metrics."""

    def __init__(self, llm_provider: LLMProvider):
        self._provider = llm_provider

    def get_model_name(self) -> str:
        return self._provider.model_string

    def load_model(self):
        return self._provider

    def generate(self, prompt: str) -> str:
        return self._provider.call_api(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
