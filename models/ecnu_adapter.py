import requests
from utils.retry import with_retry
from .base import BaseLLM

class ECNUAdapter(BaseLLM):
    def __init__(self, base_url: str, api_key: str, model: str = "ecnu-plus"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("ECNU_API_KEY is missing. Please set ECNU_API_KEY in your environment.")

        def _call():
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "You are ChatECNU"},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "model": self.model,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 401:
                raise RuntimeError("ECNU API unauthorized (401). Check your ECNU_API_KEY.")
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")

        return with_retry(_call, retries=3)
