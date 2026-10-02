import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.retry import with_retry
from .base import BaseLLM


class HuggingFaceAdapter(BaseLLM):
    def __init__(
        self,
        model: str,
        device: str = "cpu",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        token: str = "",
    ):
        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        hf_token = token or os.getenv("HF_TOKEN", "")
        self._tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token or None)
        self._model = AutoModelForCausalLM.from_pretrained(model, token=hf_token or None)
        if device != "cpu":
            self._model = self._model.to(device)

    def generate(self, prompt: str) -> str:
        def _call():
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        return with_retry(_call, retries=2)
