"""
FallbackAdapter ── 主模型网络不通时自动切换到备用模型。

用途：从国内访问 OpenRouter / Groq / Gemini 等海外 API 时，
GFW 可能导致 SSL 阻断或连接超时，此时自动 fallback 到 ECNU 等国内可用的模型。
"""

from .base import BaseLLM


def _is_network_error(exc: Exception) -> bool:
    """沿着异常链检测是否为网络连接 / 超时类错误。"""
    current: BaseException | None = exc
    while current is not None:
        name = type(current).__name__.lower()
        if any(kw in name for kw in ("connection", "connect", "timeout", "network")):
            return True
        current = current.__cause__ or current.__context__
    return False


class FallbackAdapter(BaseLLM):
    """先尝试 primary，遇到网络错误时自动切到 fallback。"""

    def __init__(self, primary: BaseLLM, fallback: BaseLLM):
        self._primary = primary
        self._fallback = fallback

    def generate(self, prompt: str) -> str:
        try:
            return self._primary.generate(prompt)
        except Exception as e:
            if _is_network_error(e):
                print(
                    f"[Fallback] 主模型网络不通 ({type(e).__name__})，"
                    f"自动切换到 fallback 模型"
                )
                return self._fallback.generate(prompt)
            # 非网络错误（例如 API key 错误、参数错误）直接抛出，不 fallback
            raise
