"""
通用 OpenAI-Compatible API 适配器。

支持任何兼容 OpenAI chat/completions 接口的服务：
- Google Gemini (via AI Studio):  https://generativelanguage.googleapis.com/v1beta/openai/
- Groq Cloud:                  https://api.groq.com/openai/v1
- DeepSeek:                    https://api.deepseek.com
- OpenRouter:                  https://openrouter.ai/api/v1
- xAI Grok:                    https://api.x.ai/v1
- 以及任何自部署的 vLLM / Ollama / LiteLLM 端点。

使用方式：
    在 .env 中设置：
    MODEL_PROVIDER=gemini          # 或 groq / deepseek / openrouter / xai
    GEMINI_API_KEY=xxx
    GEMINI_MODEL=gemini-2.5-flash

    或在 config/settings.py 中通过环境变量映射。
"""

from openai import OpenAI
from utils.retry import with_retry
from .base import BaseLLM


class OpenAICompatibleAdapter(BaseLLM):
    """通用 OpenAI-compatible 适配器，支持任意端点。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        extra_headers: dict | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        client_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"API key is missing for model '{self.model}' at {self.base_url}. "
                "Please set the corresponding API key in your environment."
            )

        def _call():
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                stream=False,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            choices = response.choices or []
            if not choices:
                return ""
            return choices[0].message.content or ""

        return with_retry(_call, retries=3)
