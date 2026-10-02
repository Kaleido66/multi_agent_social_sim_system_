from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.json_utils import parse_with_schema

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "round": {"type": "integer"},
        "overall": {"type": "string"},
        "stances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "stance": {"type": "string"},
                    "stance_label": {"type": "string"},
                    "stance_score": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["agent", "stance", "stance_label", "stance_score"],
            },
        },
    },
    "required": ["overall", "stances"],
}


class SummaryAgent(BaseAgent):
    def summarize_round(self, event_id: str, round_id: int, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = []
        for it in interactions:
            if not isinstance(it, dict):
                continue
            parsed = it.get("parsed", {})
            message = it.get("message", "")
            if isinstance(parsed, dict):
                message = parsed.get("message_zh") or parsed.get("message") or message
            normalized.append({**it, "message": message})
        prompt = (
            "你是观点总结智能体。请根据本轮发言，提炼每个智能体的明确立场，并给出整体总结。"
            "每个智能体的立场需要给出：stance（简短文字）、stance_label（支持/中立/反对）、"
            "stance_score（-1到1之间的实数）。"
            "输出必须为JSON，字段event_id/round/overall/stances。"
            f"Schema: {SUMMARY_SCHEMA}。"
            f"事件ID：{event_id}，轮次：{round_id}。本轮发言：{normalized}。"
        )
        response = self._call_llm("round_summary", prompt)
        parsed = parse_with_schema(response, SUMMARY_SCHEMA)
        if not parsed.get("valid"):
            return {
                "event_id": event_id,
                "round": round_id,
                "overall": response.strip()[:400],
                "stances": [],
                "raw": response,
            }
        summary = parsed.get("parsed") or {}
        summary.setdefault("event_id", event_id)
        summary.setdefault("round", round_id)
        return summary
