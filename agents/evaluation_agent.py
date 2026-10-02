from typing import Dict, Any
from .base_agent import BaseAgent
from utils.schemas import EVALUATION_SCHEMA
from utils.json_utils import parse_with_schema

class EvaluationAgent(BaseAgent):
    def observation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "observation": "skip"}

    def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "thought": "skip"}

    def reflect(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "reflection": "skip"}

    def _compact_eval_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        simulations = state.get("simulation_result", {}) or {}
        round_summaries = state.get("round_summaries", []) or []
        # compact summaries: keep last 5, drop long text
        compact_summaries = []
        for rs in round_summaries[-5:]:
            if not isinstance(rs, dict):
                compact_summaries.append(str(rs)[:300])
                continue
            compact_summaries.append({
                "round": rs.get("round"),
                "stances": rs.get("stances", []),
                "summary": (rs.get("summary") or "")[:300],
            })
        # compact simulations: keep event titles and rounds count
        events = []
        for evlog in simulations.get("events", []) if isinstance(simulations, dict) else []:
            ev = evlog.get("event", {}) if isinstance(evlog, dict) else {}
            rounds = evlog.get("rounds", []) if isinstance(evlog, dict) else []
            events.append({
                "id": ev.get("id"),
                "title": ev.get("title"),
                "rounds": len(rounds),
            })
        return {"events": events, "round_summaries": compact_summaries}

    def act(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        compact = self._compact_eval_state(state)
        prompt = (
            f"你是{self.persona.role}。请评估模拟结果，检测偏见、极化、信息污染，并输出风险等级、潜在问题和改进建议。"
            "如果需要，请给出persona_updates用于动态调整媒体/普通人/政策Agent。"
            "请优先依据每轮观点总结中的立场判断进行分析与评估。"
            f"只输出符合Schema的JSON（键为英文，值用中文）：{EVALUATION_SCHEMA}。"
            f"每轮观点总结：{compact.get('round_summaries')}。模拟结果摘要：{compact.get('events')}。"
        )
        response = self._call_llm("act", prompt)
        parsed = parse_with_schema(response, EVALUATION_SCHEMA) if response else {"parsed": None, "valid": False, "error": "empty_response", "raw": ""}
        self.memory.add_memory(response, "episodic", {"step": "act", "parsed": parsed})
        return {"agent": self.name, "evaluation": parsed}
