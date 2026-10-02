import random
from typing import Any, Dict, List

from agents.role_library import match_participants
from utils.control_store import get_selected_event, set_selected_event
from utils.runtime_store import runtime_store


def resolve_agent_goal(agent: Any, event: Dict[str, Any]) -> str:
    goals = event.get("agent_goals", {}) if isinstance(event, dict) else {}
    if isinstance(goals, dict):
        if agent.name in goals:
            return goals.get(agent.name, "")
        if agent.persona.role in goals:
            return goals.get(agent.persona.role, "")
    return agent.persona.goal


def bootstrap_node(graph_obj: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    model_info = state.get("model_info", {}) if isinstance(state, dict) else {}
    if not graph_obj.resume_run or not runtime_store.current_run_path:
        runtime_store.start_run(
            {
                "language_mode": graph_obj.language_mode,
                "model_info": model_info,
            }
        )
    target_event_id = state.get("event_id") or get_selected_event()
    last_event_id = runtime_store.state.get("last_event_id") if isinstance(runtime_store.state, dict) else None
    if graph_obj.reset_memory_on_run and target_event_id and target_event_id != last_event_id:
        graph_obj._clear_agent_memories()
    runtime_store.set_state(
        {
            "language_mode": graph_obj.language_mode,
            "last_event_id": target_event_id,
            "model_info": model_info,
        }
    )
    event_id = state.get("event_id")
    if event_id:
        set_selected_event(event_id)
    if state.get("case_event_only"):
        chosen_id = event_id or get_selected_event() or "evt_case_multiparty_a"
        set_selected_event(chosen_id)
    db_events = graph_obj._load_events_db()
    if state.get("case_event_only"):
        chosen_id = event_id or get_selected_event() or "evt_case_multiparty_a"
        db_events = [e for e in db_events if e.get("id") == chosen_id]
    if db_events:
        if len(db_events) > graph_obj.events_per_run:
            db_events = random.sample(db_events, graph_obj.events_per_run)
        else:
            random.shuffle(db_events)
    changed = False
    for i, e in enumerate(db_events):
        if not e.get("participant_ids"):
            e["participant_ids"] = match_participants(e, graph_obj.role_pool)
            e["event_index"] = i
            changed = True
    if changed:
        graph_obj._store_events_db(db_events)
    runtime_store.set_state({"events": db_events})
    return {"events": db_events, "start_mode": "db"}


def collect_node(graph_obj: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    graph_obj._log("[collect_agent] skipped (disabled)")
    events = graph_obj._load_events_db()
    runtime_store.set_state({"events": events})
    return {"events": events, "logs": state.get("logs", [])}


def clean_node(graph_obj: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    graph_obj._log("[clean_agent] start")
    runtime_store.set_context({"agent": "cleaner", "step": "clean"})
    result = graph_obj.cleaner.run(state)
    runtime_store.clear_context()
    graph_obj._log("[clean_agent] done")
    runtime_store.set_state({**state, "clean_data": result.get("action", {}).get("clean_data")})
    return {"clean_data": result.get("action", {}).get("clean_data"), "logs": state.get("logs", []) + [result]}


def evaluate_node(graph_obj: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    graph_obj._log("[evaluate_agent] start")
    sim_result = state.get("simulation_result")
    if not sim_result:
        sim_result = runtime_store.state.get("simulation_result") if isinstance(runtime_store.state, dict) else None
    if not sim_result:
        runtime_store.save_run(tag="empty_simulation")
        return {
            "evaluation": {"error": "empty_simulation"},
            "feedback": {},
            "logs": state.get("logs", []),
            "simulation_result": {"events": []},
        }

    round_summaries: List[Dict[str, Any]] = []
    if isinstance(sim_result, dict):
        for evlog in sim_result.get("events", []):
            if not isinstance(evlog, dict):
                continue
            rounds = evlog.get("rounds", []) if isinstance(evlog.get("rounds", []), list) else []
            for r in rounds:
                summary = r.get("summary") if isinstance(r, dict) else None
                if summary:
                    round_summaries.append(summary)

    eval_state = dict(state)
    eval_state["simulation_result"] = sim_result
    eval_state["round_summaries"] = round_summaries
    runtime_store.set_context({"agent": "evaluator", "step": "evaluate"})
    result = graph_obj.evaluator.run(eval_state)
    runtime_store.clear_context()

    evaluation = result.get("action", {}).get("evaluation")
    feedback = {}
    if isinstance(evaluation, dict):
        parsed = evaluation.get("parsed")
        if isinstance(parsed, dict):
            feedback = parsed.get("feedback", {})
            persona_updates = parsed.get("persona_updates", [])
            if isinstance(persona_updates, list):
                for upd in persona_updates:
                    agent_name = upd.get("agent")
                    if agent_name == "media":
                        graph_obj.media_agent.update_persona(upd)
                    elif agent_name == "citizen":
                        graph_obj.citizen_agent.update_persona(upd)
                    elif agent_name == "policy":
                        graph_obj.policy_agent.update_persona(upd)
    runtime_store.save_run(tag="ok")
    graph_obj._log("[evaluate_agent] done")

    events = state.get("events", [])
    if not events and isinstance(sim_result, dict):
        events = [e.get("event") for e in sim_result.get("events", []) if isinstance(e, dict) and e.get("event")]
    return {
        "evaluation": evaluation,
        "feedback": feedback,
        "logs": state.get("logs", []) + [result],
        "simulation_result": sim_result,
        "events": events,
    }
