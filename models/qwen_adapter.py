from .base import BaseLLM

class QwenAdapter(BaseLLM):
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("QwenAdapter is a placeholder. Implement API integration here.")
