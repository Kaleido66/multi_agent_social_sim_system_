from openai import OpenAI
from utils.retry import with_retry
from .base import BaseLLM


class GiteeOpenAIAdapter(BaseLLM):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "MiniMax-M2.5",
        top_k: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.7,
        frequency_penalty: float = 1.0,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"X-Failover-Enabled": "true"},
        )

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GITEE_OPENAI_API_KEY is missing. Please set GITEE_OPENAI_API_KEY in your environment.")

        def _call():
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful and harmless assistant."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                stream=False,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={"top_k": self.top_k},
                frequency_penalty=self.frequency_penalty,
            )
            choices = response.choices or []
            if not choices:
                return ""
            return choices[0].message.content or ""

        return with_retry(_call, retries=3)
