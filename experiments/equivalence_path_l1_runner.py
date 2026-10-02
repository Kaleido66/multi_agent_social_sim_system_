#!/usr/bin/env python3
"""
Equivalence Path L1 runner (stateless version).

Does NOT modify any file under the main framework (agents/, graph/, utils/).
Only reuses read-only imports: persona data (BASE_ROLES), the event DB, the
model factory, and the stance-evaluation pipeline.

Design, confirmed with the researcher:
  - Each run is a single, independent, stateless call. No multi-round
    conversation history is carried between or within runs. Because the
    backbone models used here are stateless, "20 independent stateless calls"
    is the correct operationalization of "framework-free" behavior, not
    "one 20-round stateful conversation".
  - arm0_no_persona: no persona/system-prompt content at all; the model is
    asked the core question directly.
  - arm1_persona: the model is given the persona's full identity description
    (rendered as natural-language Chinese, matching the production framework's
    style) and nothing else (no tools, no other agents, no induction, no
    constraints, no memory, no social-state summary).
  - All 11 personas in BASE_ROLES are covered.
  - Each (persona, condition) pair is run independently under each of the
    four backbone models in MODEL_PRESETS.

The natural-language persona rendering in `render_persona_narrative()` and
the core-question template in `CORE_QUESTION_TEMPLATE` were both verified
against a real production log provided by the researcher (not reconstructed
from source-code inspection alone), including field order, phrasing, and the
two fixed-format instructions (<=40-character stance sentence; <=200-character
overall answer, itemized). agents/social_agents.py itself is not imported
from, referenced, or modified by this script.

*** OPEN ITEM: stance-evaluation input-format compatibility, unverified ***
In production, evaluate_agent_stance() scores a structured `message` field
that decide() guarantees contains "观点:"/"论据:" markers (see the fallback
logic there). In this experiment, arm0/arm1 responses instead follow
CORE_QUESTION_TEMPLATE's own format (a short stance sentence + itemized
answer), which will NOT contain those markers. Whether
utils/stance_eval.evaluate_agent_stance() degrades gracefully, silently
scores worse, or behaves identically regardless of input format has not been
verified — read utils/stance_eval.py before trusting real (non-dry-run)
scores from this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.role_library import BASE_ROLES  # noqa: E402
from config.settings import load_config  # noqa: E402
from graph.io_helpers import load_events_db  # noqa: E402
from utils.stance_eval import evaluate_agent_stance  # noqa: E402


EVENT_ID = "evt_case_multiparty_0423"
DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "experiments", "equivalence_path_l1")

# Four backbone models for this supplementary experiment (a deliberate subset
# of the six models used in the main cross-model validation). `model` is the
# string actually passed to the API client; `display` is the name used in all
# output records/plots/CSVs so that downstream analysis lines up with the
# main experiment's model naming (e.g. in data/plots/ablation_2x2_longtable.csv).
#
# ecnu-plus / ecnu-max are internal gateway aliases that must be used for the
# actual API call (calling with the "real" model name fails against this
# infrastructure); they route to Qwen3.6-27B and DeepSeek-V4-Flash respectively.
MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "ecnu-plus": {"provider": "ecnu", "model": "ecnu-plus", "display": "qwen3.6-27b"},
    "ecnu-max": {"provider": "ecnu", "model": "ecnu-max", "display": "deepseek-v4-flash"},
    "llama-3.3-70b": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct", "display": "llama-3.3-70b"},
    "claude-sonnet-4.6": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6", "display": "claude-sonnet-4.6"},
    "gpt-5.6-terra": {"provider": "openrouter", "model": "openai/gpt-5.6-terra", "display": "gpt-5.6-terra"},
    "gemini-3-flash": {"provider": "openrouter", "model": "google/gemini-3-flash-preview", "display": "gemini-3-flash"},
}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    model: str
    display: str


def model_specs(names: Iterable[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name == "all":
            return [ModelSpec(alias=k, provider=v["provider"], model=v["model"], display=v["display"])
                    for k, v in MODEL_PRESETS.items()]
        if name not in MODEL_PRESETS:
            allowed = ", ".join(sorted(MODEL_PRESETS))
            raise ValueError(f"Unknown model preset '{name}'. Allowed: {allowed}, all")
        spec = MODEL_PRESETS[name]
        specs.append(ModelSpec(alias=name, provider=spec["provider"], model=spec["model"], display=spec["display"]))
    return specs


# ---------------------------------------------------------------------------
# Persona rendering (duplicated here on purpose, NOT imported from
# agents/social_agents.py, and agents/social_agents.py is not touched).
# ---------------------------------------------------------------------------

# Verified against a real production log provided by the researcher on
# 2026-09-14. Field order and phrasing below are confirmed accurate; only
# the values are substituted per-persona from BASE_ROLES.
def render_persona_narrative(role: Dict[str, Any]) -> str:
    """
    Render only the persona-identity portion of an agent's prompt, as
    natural-language Chinese text — matching the production framework's
    actual prompt structure (confirmed against a real log, not reconstructed
    from source inspection). Excludes goals, tools, induction material,
    other agents' views, memory retrieval, and social-state summaries;
    those are precisely the "framework" components this experiment is
    designed to omit.

    `role` is one entry from agents.role_library.BASE_ROLES (a dict), used
    as-is (values copied verbatim, not paraphrased).
    """
    def g(key: str, default: str = "") -> Any:
        return role.get(key, default)

    lines = [
        f"你是{g('role')}，姓名{g('name')}，国籍{g('nationality')}，"
        f"性别{g('gender')}，年龄{g('age')}。",
        f"政治身份：{g('political_identity')}，阵营：{g('faction')}。",
        f"政体立场：{g('regime_stance')}。",
        f"权力偏好：{g('authority_preference')}；合法性来源：{g('legitimacy_source')}；"
        f"分配倾向：{g('distribution_preference')}。",
        f"价值观：{g('values')}。",
        f"信念：{g('belief')}。",
        f"政治观点：{g('polity_view')}。",
        f"个人属性：收入{g('income_level')}，职业类型{g('occupation_type')}，工作强度{g('work_intensity')}。",
        f"家庭属性：{'有子女' if g('has_children') else '无子女'}，"
        f"教育焦虑{g('education_anxiety')}，家庭负担{g('family_burden')}。",
        f"价值权重：{g('value_weights')}。",
    ]
    return "\n".join(lines)


def build_persona_system_prompt(role: Dict[str, Any]) -> str:
    narrative = render_persona_narrative(role)
    return (
        "你正在进行单智能体独立对话实验。每一次调用都是全新、独立的对话，"
        "不携带此前任何交流记录，也不存在其他智能体、工具调用、诱导信息、"
        "约束材料或社会状态。请仅依据下列角色设定，以第一人称回答接下来提出的问题。\n"
        f"{narrative}"
    )


def build_raw_system_prompt() -> str:
    return (
        "你是一个通用助手。这是一次独立、无历史记录的对话，"
        "不扮演任何特定人物，不带有任何角色设定。请直接回答问题。"
    )


# ---------------------------------------------------------------------------
# Core question
# ---------------------------------------------------------------------------

# Fixed template, confirmed against a real production log: does not vary by
# event or persona. `{title}` and `{content}` are the only substituted
# values. Note `{content}` already carries its own trailing punctuation
# (typically "？"), so no separator is inserted before "请将回答控制在...".
CORE_QUESTION_TEMPLATE = (
    "请基于你的价值观与国家体制认知，给出一句明确立场（不超过40字，可带情绪色彩）。"
    "事件：{title}。内容：{content}"
    "请将回答控制在不超过200字以内，尽量精炼、分点陈述。"
)


def load_event(event_id: str = EVENT_ID) -> Dict[str, str]:
    for event in load_events_db():
        if event.get("id") == event_id:
            title = str(event.get("title") or "")
            content = str(event.get("content") or "")
            if not title or not content:
                raise RuntimeError(f"Event {event_id} is missing title and/or content.")
            return {"title": title, "content": content}
    raise RuntimeError(f"Event {event_id} not found in data/events.jsonl.")


def build_core_question_text(event: Dict[str, str]) -> str:
    return CORE_QUESTION_TEMPLATE.format(title=event["title"], content=event["content"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def stance_score_from_eval(result: Dict[str, Any]) -> Optional[float]:
    try:
        return max(-1.0, min(1.0, float(result.get("support", 0.0)) - float(result.get("oppose", 0.0))))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single stateless call
# ---------------------------------------------------------------------------

def single_call_prompt(question_text: str) -> str:
    # question_text is already the fully-formed CORE_QUESTION_TEMPLATE
    # output (title + content + both fixed-format instructions), so it is
    # sent as-is with no additional wrapping preamble.
    return question_text


def set_adapter_system_prompt(llm: Any, system_prompt: str) -> None:
    if hasattr(llm, "system_prompt"):
        try:
            llm.system_prompt = system_prompt
        except Exception:
            pass


def generate_single_response(llm: Any, system_prompt: str, question: str) -> Dict[str, str]:
    set_adapter_system_prompt(llm, system_prompt)
    prompt = single_call_prompt(question)
    if not hasattr(llm, "system_prompt"):
        prompt = f"系统设定：\n{system_prompt}\n\n{prompt}"
    return {"prompt": prompt, "response": llm.generate(prompt)}


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one(
    *,
    llm: Any,
    eval_llm: Any,
    model_spec: ModelSpec,
    role: Optional[Dict[str, Any]],
    condition: str,
    run_id: int,
    event: Dict[str, str],
    question_text: str,
    out_jsonl: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    persona_id = str(role.get("id")) if role else ""
    persona_name = str(role.get("name")) if role else ""
    system_prompt = build_raw_system_prompt() if condition == "arm0_no_persona" else build_persona_system_prompt(role)

    if dry_run:
        raw_response = f"DRY_RUN_RESPONSE {condition} {persona_id or 'none'} run={run_id}"
        prompt = single_call_prompt(question_text)
    else:
        generated = generate_single_response(llm, system_prompt, question_text)
        prompt = generated["prompt"]
        raw_response = generated["response"]

    eval_result = evaluate_agent_stance(eval_llm, persona_name or "no_persona", raw_response, language_mode="zh")
    stance_score = stance_score_from_eval(eval_result)

    record = {
        "persona_id": persona_id,
        "persona_name": persona_name,
        "condition": condition,
        "run_id": run_id,
        "event_id": EVENT_ID,
        "event_title": event["title"],
        "event_content": event["content"],
        "question_text": question_text,
        "model_alias": model_spec.display,     # display name, aligned with main-experiment naming
        "model_call_name": model_spec.model,   # actual name/alias passed to the API client
        "model_provider": model_spec.provider,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "raw_response": raw_response,
        "stance_eval": eval_result,
        "stance_score": stance_score,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_jsonl(out_jsonl, record)
    return record


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def save_methods_manifest(out_dir: str, event: Dict[str, str], question_text: str) -> str:
    path = os.path.join(out_dir, "methods_manifest.json")
    payload = {
        "design": "stateless_independent_calls",
        "design_note": (
            "Each run is one independent, stateless API call; no conversation "
            "history is carried across runs or within a run (backbone models "
            "are stateless, so this is the correct operationalization of "
            "'framework-free' behavior)."
        ),
        "event_id": EVENT_ID,
        "event_title": event["title"],
        "event_content": event["content"],
        "core_question_template": CORE_QUESTION_TEMPLATE,
        "core_question_rendered": question_text,
        "core_question_source": "data/events.jsonl title+content fields (loaded via graph.io_helpers.load_events_db)",
        "core_question_template_note": (
            "Fixed template confirmed against a real production log; does not "
            "vary by event or persona. Two format constraints: <=40-character "
            "stance sentence, <=200-character itemized overall answer."
        ),
        "persona_source": "agents/role_library.py BASE_ROLES (values used verbatim)",
        "persona_rendering": (
            "Natural-language rendering in this script's render_persona_narrative(), "
            "verified field-order-and-phrasing against a real production log "
            "provided by the researcher on 2026-09-14. NOT imported from "
            "agents/social_agents.py; agents/social_agents.py was not modified."
        ),
        "stance_eval_source": "utils/stance_eval.py evaluate_agent_stance; stance_score = support - oppose",
        "stance_eval_input_format_caveat": (
            "Production evaluate_agent_stance() normally scores a structured "
            "message containing '观点:'/'论据:' markers. Arm0/arm1 responses "
            "here follow CORE_QUESTION_TEMPLATE's own format instead and will "
            "not contain those markers. Compatibility with the scorer has not "
            "been verified; check utils/stance_eval.py before trusting scores."
        ),
        "persona_count": len(BASE_ROLES),
        "models": {k: v["display"] for k, v in MODEL_PRESETS.items()},
        "conditions": ["arm0_no_persona", "arm1_persona"],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L1 equivalence-path stateless baselines.")
    parser.add_argument("--models", default="all", help="Comma-separated presets or 'all'.")
    parser.add_argument("--runs", type=int, default=10, help="Independent stateless calls per (model, condition, persona).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--eval-provider", default="", help="Override evaluator provider; default follows each model.")
    parser.add_argument("--eval-model", default="", help="Override evaluator model name.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call generation LLMs; useful for tests/smoke checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    specs = model_specs(args.models.split(","))
    event = load_event(EVENT_ID)
    question_text = build_core_question_text(event)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(args.out_dir, f"l1_equivalence_raw_{run_ts}.jsonl")
    save_methods_manifest(args.out_dir, event, question_text)

    for spec in specs:
        print(f"[model] {spec.alias} -> display={spec.display} provider={spec.provider} model={spec.model}")
        eval_provider = args.eval_provider or spec.provider
        eval_model = args.eval_model or spec.model
        llm = None
        eval_llm = None
        if not args.dry_run:
            from main import _build_llm, _resolve_model_params

            llm = _build_llm(cfg, spec.provider, explicit_model=spec.model)
            eval_llm = _build_llm(cfg, eval_provider, explicit_model=eval_model)
            params = _resolve_model_params(cfg, spec.provider, explicit_model=spec.model)
            if float(params.get("temperature", 0.7) or 0.7) == 0.0:
                raise RuntimeError("Temperature must not be 0 for this experiment.")

        # --- arm0_no_persona: persona-independent, run once per model ---
        for run_id in range(1, args.runs + 1):
            print(f"  arm0_no_persona run={run_id}/{args.runs}")
            run_one(
                llm=llm, eval_llm=eval_llm, model_spec=spec, role=None,
                condition="arm0_no_persona", run_id=run_id, event=event,
                question_text=question_text, out_jsonl=out_jsonl, dry_run=args.dry_run,
            )

        # --- arm1_persona: all 11 personas ---
        for role in BASE_ROLES:
            for run_id in range(1, args.runs + 1):
                print(f"  arm1_persona {role.get('id')} run={run_id}/{args.runs}")
                run_one(
                    llm=llm, eval_llm=eval_llm, model_spec=spec, role=role,
                    condition="arm1_persona", run_id=run_id, event=event,
                    question_text=question_text, out_jsonl=out_jsonl, dry_run=args.dry_run,
                )

    print(f"[done] JSONL: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())