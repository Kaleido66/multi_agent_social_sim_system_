import json
import os
from typing import Any, Dict, List

REGISTRY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_registry.json"))


def save_registry(registry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(registry, ensure_ascii=False, indent=2))


def load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def register_agents(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    registry = {a["id"]: a for a in agents}
    save_registry(registry)
    return registry
