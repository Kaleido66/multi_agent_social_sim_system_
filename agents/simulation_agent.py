from typing import Dict, Any
from .base_agent import BaseAgent
from utils.schemas import SIMULATION_SCHEMA
from utils.json_utils import parse_with_schema

class SimulationAgent(BaseAgent):
    def action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = state.get("cleaned_data", {})
        persona = self.persona or "neutral citizen"

        if self.llm is None:
            simulation_result = {
                "persona": persona,
                "reaction": "concerned",
                "signal": "possible value conflict",
                "rationale": "fallback rule-based response",
                "input": cleaned,
            }
            return {"agent": self.name, "simulation": simulation_result}

        system_prompt = (
            f"You are role-playing as: {persona}. "
            "Given the cleaned data, infer possible social reactions, value conflicts, "
            "and risk signals. Return ONLY JSON that matches this schema: "
            f"{SIMULATION_SCHEMA}"
        )
        user_prompt = f"Cleaned data: {cleaned}"
        prompt = f"SYSTEM: {system_prompt}\nUSER: {user_prompt}"

        response = self.llm.generate(prompt)
        parsed = parse_with_schema(response, SIMULATION_SCHEMA)

        simulation_result = {
            "persona": persona,
            "parsed": parsed,
            "input": cleaned,
        }
        return {"agent": self.name, "simulation": simulation_result}
