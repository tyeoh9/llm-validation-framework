import litellm

'''
To see which providers this class supports, visit https://docs.litellm.ai/docs/providers
'''

class LLMProvider:
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