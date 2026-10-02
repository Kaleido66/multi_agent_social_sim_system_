import requests
from utils.retry import with_retry
from .base import BaseLLM

class OllamaAdapter(BaseLLM):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        def _call():
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        return with_retry(_call, retries=3)
