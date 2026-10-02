from .ollama_adapter import OllamaAdapter
from .ecnu_adapter import ECNUAdapter
from .qwen_adapter import QwenAdapter
from .minimax_adapter import MinimaxAdapter
from .gitee_adapter import GiteeAdapter
from .gitee_openai_adapter import GiteeOpenAIAdapter
from .hf_adapter import HuggingFaceAdapter
from .openai_compatible_adapter import OpenAICompatibleAdapter
from .base import BaseLLM


# ---------------------------------------------------------------------------
# Rate-limited adapter wrapper
# ---------------------------------------------------------------------------

class RateLimitedAdapter(BaseLLM):
    """Transparent rate-limiting proxy around any :class:`BaseLLM` adapter."""

    def __init__(self, adapter: BaseLLM, rate_limiter):
        self._adapter = adapter
        self._rate_limiter = rate_limiter

    def generate(self, prompt: str) -> str:
        self._rate_limiter.acquire()
        return self._adapter.generate(prompt)

    # forward other commonly-accessed attributes to the wrapped adapter
    def __getattr__(self, name):
        return getattr(self._adapter, name)

# ---------------------------------------------------------------------------
# 国际基座模型 预置端点（免费 / 极低成本渠道）
# ---------------------------------------------------------------------------
# Google AI Studio（免费，无需绑卡）: https://aistudio.google.com/apikey
# Groq Cloud（免费，额度极高）:       https://console.groq.com
# DeepSeek（极便宜）:                https://platform.deepseek.com
# OpenRouter（聚合，有免费模型）:     https://openrouter.ai
# xAI Grok（需付费）:                https://x.ai/api
# ---------------------------------------------------------------------------

# 每个 provider 可用的模型示例（仅作文档参考，不自动兜底）：
#   gemini    → gemini-2.5-flash / gemini-2.5-pro
#   groq      → llama-3.3-70b-versatile / mixtral-8x7b-32768
#   deepseek  → deepseek-chat / deepseek-reasoner
#   openrouter→ google/gemini-2.5-flash / meta-llama/llama-4-maverick
#   xai       → grok-2 / grok-2-latest
INTERNATIONAL_PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "model_env": "XAI_MODEL",
    },
}


class LLMFactory:
    # Adapter types / providers that make remote API calls → should be rate-limited.
    _LOCAL_PROVIDERS = frozenset({"ollama", "hf", "huggingface", "qwen", "minimax"})

    @staticmethod
    def create(provider: str, rate_limiter=None, **kwargs):
        adapter = LLMFactory._create_raw(provider, **kwargs)
        if rate_limiter is not None and (provider or "").lower() not in LLMFactory._LOCAL_PROVIDERS:
            return RateLimitedAdapter(adapter, rate_limiter)
        return adapter

    @staticmethod
    def _create_raw(provider: str, **kwargs):
        provider = (provider or "").lower()

        # ── 国际基座模型（OpenAI-compatible 通用） ──
        if provider in INTERNATIONAL_PRESETS:
            import os
            preset = INTERNATIONAL_PRESETS[provider]
            base_url = kwargs.get("base_url") or preset["base_url"]
            api_key = kwargs.get("api_key") or os.getenv(preset["api_key_env"], "")
            model = kwargs.get("model") or os.getenv(preset["model_env"], "")
            if not model:
                raise RuntimeError(
                    f"Model not specified for provider '{provider}'. "
                    f"Set {preset['model_env']} in .env or pass model= explicitly."
                )
            return OpenAICompatibleAdapter(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )

        # ── 通用 OpenAI-compatible（手动指定 base_url） ──
        if provider in ["openai_compatible", "generic_openai", "custom_openai"]:
            return OpenAICompatibleAdapter(
                base_url=kwargs["base_url"],
                api_key=kwargs.get("api_key", ""),
                model=kwargs.get("model", "default"),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )

        # ── 已有适配器 ──
        if provider == "ollama":
            return OllamaAdapter(kwargs["base_url"], kwargs["model"])
        if provider == "ecnu":
            return ECNUAdapter(kwargs["base_url"], kwargs["api_key"], kwargs["model"])
        if provider == "qwen":
            return QwenAdapter()
        if provider == "minimax":
            return MinimaxAdapter()
        if provider == "gitee":
            return GiteeAdapter(
                kwargs["base_url"],
                kwargs["api_key"],
                kwargs["model"],
                think=kwargs.get("think", False),
                top_k=kwargs.get("top_k", 50),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.7),
                frequency_penalty=kwargs.get("frequency_penalty", 1.0),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        if provider in ["gitee_openai", "gitee-minimax", "gitee_openai_minimax"]:
            return GiteeOpenAIAdapter(
                kwargs["base_url"],
                kwargs["api_key"],
                kwargs["model"],
                top_k=kwargs.get("top_k", 50),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.7),
                frequency_penalty=kwargs.get("frequency_penalty", 1.0),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        if provider in ["gitee_openai_qwen", "gitee-qwen", "gitee_openai_qwen"]:
            return GiteeOpenAIAdapter(
                kwargs["base_url"],
                kwargs["api_key"],
                kwargs["model"],
                top_k=kwargs.get("top_k", 50),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.7),
                frequency_penalty=kwargs.get("frequency_penalty", 1.0),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        if provider in ["hf", "huggingface"]:
            return HuggingFaceAdapter(
                model=kwargs["model"],
                device=kwargs.get("device", "cpu"),
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                token=kwargs.get("token", ""),
            )
        raise ValueError(f"Unknown provider: {provider}")
