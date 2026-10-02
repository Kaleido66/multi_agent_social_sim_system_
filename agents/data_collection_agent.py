from typing import Dict, Any
from .base_agent import BaseAgent
from utils.schemas import COLLECTION_SCHEMA
from utils.json_utils import parse_with_schema

class DataCollectionAgent(BaseAgent):
    def observation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "observation": "skip"}

    def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "thought": "skip"}

    def reflect(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "reflection": "skip"}

    def act(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        problem = state.get("problem", "近一周国际热门时事")
        feedback = state.get("feedback", {})
        prompt = (
            f"你是{self.persona.role}。请生成更精准的数据采集查询和焦点。"
            f"只输出符合Schema的JSON（键为英文，值用中文）：{COLLECTION_SCHEMA}。"
            f"问题：{problem}。反馈：{feedback}。"
        )
        response = self._call_llm("act", prompt)
        parsed = parse_with_schema(response, COLLECTION_SCHEMA)
        self.memory.add_memory(response, "episodic", {"step": "act", "parsed": parsed})

        refined = problem
        if parsed.get("valid") and parsed.get("parsed"):
            refined = parsed["parsed"].get("refined_query", problem)

        skill_result = None
        if hasattr(self.skills, "call"):
            try:
                skill_result = self.skills.call("search", refined)
            except Exception:
                skill_result = None

        return {
            "agent": self.name,
            "plan": parsed,
            "raw_data": {"query": refined, "plan": parsed, "skill_result": skill_result},
        }
