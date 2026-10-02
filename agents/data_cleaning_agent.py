from typing import Dict, Any
from .base_agent import BaseAgent

class DataCleaningAgent(BaseAgent):
    def observation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "observation": "skip"}

    def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "thought": "skip"}

    def reflect(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "reflection": "skip"}

    def act(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        raw = state.get("raw_data", {})
        prompt = (
            f"你是{self.persona.role}。请对原始数据进行清洗、去重、格式对齐，并输出标准化JSON。"
            f"原始数据：{raw}。"
        )
        response = self._call_llm("act", prompt)
        self.memory.add_memory(response, "episodic", {"step": "act"})
        return {"agent": self.name, "clean_data": {"raw": raw, "llm_clean": response}}
