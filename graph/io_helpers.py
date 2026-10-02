import json
import os
from typing import Any, Dict, List


def register_memories(graph_obj: Any) -> None:
    graph_obj.memory_hub.register("collector", graph_obj.collector.memory)
    graph_obj.memory_hub.register("cleaner", graph_obj.cleaner.memory)
    graph_obj.memory_hub.register("media", graph_obj.media_agent.memory)
    graph_obj.memory_hub.register("citizen", graph_obj.citizen_agent.memory)
    graph_obj.memory_hub.register("policy", graph_obj.policy_agent.memory)
    graph_obj.memory_hub.register("evaluator", graph_obj.evaluator.memory)


def clear_agent_memories(graph_obj: Any) -> None:
    targets = [
        graph_obj.collector,
        graph_obj.cleaner,
        graph_obj.media_agent,
        graph_obj.citizen_agent,
        graph_obj.policy_agent,
        graph_obj.evaluator,
        graph_obj.summary_agent,
    ]
    for agent in targets:
        mem = getattr(agent, "memory", None)
        if mem and hasattr(mem, "clear"):
            try:
                mem.clear(graph_obj.memory_clear_types)
            except Exception:
                continue


def events_db_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "events.jsonl"))


def load_events_db() -> List[Dict[str, Any]]:
    path = events_db_path()
    if not os.path.exists(path):
        return []
    events: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return events


def store_events_db(events: List[Dict[str, Any]]) -> None:
    path = events_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in events or []:
            if isinstance(item, dict):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
