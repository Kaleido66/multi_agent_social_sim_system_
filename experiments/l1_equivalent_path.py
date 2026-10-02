#!/usr/bin/env python3
"""
L1: Equivalent Path Baseline Experiment
=========================================

Purpose
-------
Establish the baseline for the "equivalent path" question:
  - When an LLM answers a political question WITHOUT any role/framework,
    what stance does it take?  (raw baseline)
  - When the SAME LLM is given a persona description (but no multi-agent
    framework), how much does its stance shift?  (role effect)
  - Comparing these two to the full simulation results lets us decompose
    stance into:  [model prior] + [role prompt effect] + [framework effect].

Conditions
----------
  raw        No persona.  Direct question.  System prompt = "You are ChatECNU".
  raw_role   Persona header prepended.  Same question.  One run per role.
  raw_cstr   Persona header + progressive constraints (R1/R10/R18).  Same LLM call.

Outputs
-------
  data/experiments/l1_raw_stances.jsonl   – one line per raw repetition
  data/experiments/l1_role_stances.csv    – one row per (role, repetition)
  data/experiments/l1_summary.json        – aggregated means, SDs, CIs

Usage
-----
  python experiments/l1_equivalent_path.py
  python experiments/l1_equivalent_path.py --raw-repeats 20 --role-repeats 5
  python experiments/l1_equivalent_path.py --temperature 0.0   # deterministic
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -- project imports -----------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config.settings import load_config, AppConfig      # noqa: E402
from config.sim_settings import load_sim_config          # noqa: E402
from agents.role_library import BASE_ROLES               # noqa: E402  (11 roles)
from models.registry import LLMFactory                   # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

CORE_QUESTION_BASE = (
    "什么情况下多党竞争选举民主制是一个国家政体的最优解？"
)

# progressive constraint texts (hand-curated from event_constraints/evt_case_multiparty_0423.json)
CONSTRAINT_ROUNDS: Dict[str, str] = {
    "R1_baseline": (
        "什么情况下多党竞争选举民主制是一个国家政体的最优解？"
    ),
    "R2_population": (
        "当一个国家人口超过10亿，这种条件下多党竞争选举民主制是一个国家政体的最优解吗？"
    ),
    "R10_transition": (
        "当一个国家人口超过10亿，国土面积很大超过900万平方公里，且边界线很长，"
        "邻国众多且存在不止一个强国，含边疆、高原、海岛等战略要地，"
        "处于城镇化、老龄化、数字化快速推进的转型期，"
        "这种条件下多党竞争选举民主制是一个国家政体的最优解吗？"
    ),
    "R18_full": (
        "当一个国家人口超过10亿，国土面积很大超过900万平方公里，且边界线很长，"
        "邻国众多且存在不止一个强国，含边疆、高原、海岛等战略要地，"
        "有中央集权历史惯性，包含工人、农民、企业家、知识分子、新业态从业者等数十个社会阶层，"
        "处于城镇化、老龄化、数字化快速推进的转型期，"
        "有长期传统计划经济体制历史，其政治制度与社会结构、制度框架和组织方式在后续改革中仍有延续性影响；"
        "这种条件下多党竞争选举民主制是一个国家政体的最优解吗？"
    ),
}

OUT_DIR = os.path.join(ROOT, "data", "experiments")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_llm(cfg: AppConfig, provider: str, explicit_model: str = ""):
    """Mirrors main._build_llm without modifying any existing code."""
    provider = (provider or "ecnu").lower()
    if provider in ("gitee_openai_qwen", "gitee-qwen"):
        params = {
            "model": explicit_model or cfg.gitee_openai_qwen_model,
            "base_url": cfg.gitee_openai_base_url,
            "api_key": cfg.gitee_openai_qwen_api_key,
            "temperature": cfg.gitee_openai_qwen_temperature,
            "top_p": cfg.gitee_openai_qwen_top_p,
            "max_tokens": cfg.gitee_openai_qwen_max_tokens,
        }
    elif provider in ("gitee_openai", "gitee-minimax", "gitee_openai_minimax"):
        params = {
            "model": explicit_model or cfg.gitee_openai_model,
            "base_url": cfg.gitee_openai_base_url,
            "api_key": cfg.gitee_openai_api_key,
            "temperature": cfg.gitee_openai_temperature,
            "top_p": cfg.gitee_openai_top_p,
            "max_tokens": cfg.gitee_openai_max_tokens,
        }
    elif provider == "gitee":
        params = {
            "model": explicit_model or cfg.gitee_model,
            "base_url": cfg.gitee_base_url,
            "api_key": cfg.gitee_api_key,
            "temperature": cfg.gitee_temperature,
            "top_p": cfg.gitee_top_p,
            "max_tokens": cfg.gitee_max_tokens,
        }
    else:
        params = {
            "model": explicit_model or cfg.ecnu_model,
            "base_url": cfg.ecnu_base_url,
            "api_key": cfg.ecnu_api_key,
            "temperature": cfg.gitee_temperature,
            "top_p": cfg.gitee_top_p,
            "max_tokens": cfg.gitee_max_tokens,
        }
    return LLMFactory.create(
        provider,
        base_url=params.get("base_url", ""),
        model=params.get("model", ""),
        api_key=params.get("api_key", ""),
        think=False,
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 0.7),
        max_tokens=params.get("max_tokens", 1024),
    )


def build_persona_header(role: Dict[str, Any]) -> str:
    """Replicate the persona header format from base_agent._persona_header()."""
    has_children = bool(role.get("has_children", False))
    return (
        f"你是{role.get('role','')}，姓名{role.get('name','')}，"
        f"性别{role.get('gender','')}，年龄{role.get('age','')}。"
        f"政治身份：{role.get('political_identity','')}，阵营：{role.get('faction','')}。"
        f"政体立场：{role.get('regime_stance','')}；"
        f"权力偏好：{role.get('authority_preference','')}；"
        f"合法性来源：{role.get('legitimacy_source','')}；"
        f"分配倾向：{role.get('distribution_preference','')}。"
        f"价值观：{role.get('values','')}。信念：{role.get('belief','')}。"
        f"政治观点：{role.get('polity_view','')}。"
        f"个人属性：收入{role.get('income_level','')}，"
        f"职业类型{role.get('occupation_type','')}，"
        f"工作强度{role.get('work_intensity','')}。"
        f"家庭属性：{'有子女' if has_children else '无子女'}，"
        f"教育焦虑{role.get('education_anxiety','')}，"
        f"家庭负担{role.get('family_burden','')}。"
        f"价值权重：{json.dumps(role.get('value_weights',{}), ensure_ascii=False)}。"
    )


def build_prompt(question: str, persona_header: str = "") -> str:
    """Build a prompt that asks for a stance score in parseable JSON."""
    persona_block = f"{persona_header}\n\n" if persona_header else ""
    return (
        f"{persona_block}"
        f"请基于你的价值观与知识，回答以下问题。\n"
        f"问题：{question}\n\n"
        f"请严格按以下JSON格式输出（不要输出其他内容）：\n"
        f'{{"stance_score": <浮点数，范围-1到1，-1=完全反对多党竞争选举民主制，'
        f'0=中立，+1=完全支持>，'
        f'"reasoning": "<简短论证，不超过80字>"}}'
    )


def parse_stance(text: str) -> Optional[float]:
    """Extract stance_score from LLM response.  Returns None on failure."""
    if not text:
        return None
    # try direct JSON parse
    try:
        obj = json.loads(text)
        return float(obj.get("stance_score", obj.get("score", 0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # try to find JSON block
    m = re.search(r"\{[^{}]*\"stance_score\"\s*:\s*(-?[\d.]+)[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # try looser regex
    m = re.search(r"stance_score[:\s]*(-?[\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # try score field
    m = re.search(r"\"score\"\s*:\s*(-?[\d.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# core experiment
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    condition: str       # "raw" | "raw_role" | "raw_cstr"
    constraint_label: str  # "R1_baseline" | "R18_full" | ...
    agent_id: str        # "" for raw
    agent_name: str
    nationality: str
    faction: str
    repeat: int
    stance_score: Optional[float]
    raw_response: str
    timestamp: str


def run_raw_baseline(
    llm,
    temperature: float,
    repeats: int,
    constraint_labels: List[str],
) -> List[RunRecord]:
    """Run the raw (no-persona) baseline for each constraint level."""
    records: List[RunRecord] = []
    for clabel in constraint_labels:
        question = CONSTRAINT_ROUNDS[clabel]
        prompt = build_prompt(question, persona_header="")
        for rep in range(1, repeats + 1):
            print(f"  [raw] {clabel} repeat={rep}/{repeats}")
            try:
                raw = llm.generate(prompt)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                raw = ""
            stance = parse_stance(raw)
            records.append(RunRecord(
                condition="raw", constraint_label=clabel,
                agent_id="", agent_name="(none)", nationality="", faction="",
                repeat=rep, stance_score=stance, raw_response=raw,
                timestamp=now_iso(),
            ))
    return records


def run_role_baseline(
    llm,
    roles: List[Dict[str, Any]],
    repeats: int,
    constraint_labels: List[str],
) -> List[RunRecord]:
    """Run the role-prompted (no framework) condition for each role × constraint."""
    records: List[RunRecord] = []
    total = len(roles) * len(constraint_labels) * repeats
    idx = 0
    for role in roles:
        persona = build_persona_header(role)
        for clabel in constraint_labels:
            question = CONSTRAINT_ROUNDS[clabel]
            prompt = build_prompt(question, persona_header=persona)
            for rep in range(1, repeats + 1):
                idx += 1
                print(f"  [raw_role] {role['name']}({role['id']}) {clabel} "
                      f"repeat={rep}/{repeats}  [{idx}/{total}]")
                try:
                    raw = llm.generate(prompt)
                except Exception as exc:
                    print(f"    ERROR: {exc}")
                    raw = ""
                stance = parse_stance(raw)
                records.append(RunRecord(
                    condition="raw_role", constraint_label=clabel,
                    agent_id=str(role.get("id", "")),
                    agent_name=str(role.get("name", "")),
                    nationality=str(role.get("nationality", "")),
                    faction=str(role.get("faction", "")),
                    repeat=rep, stance_score=stance, raw_response=raw,
                    timestamp=now_iso(),
                ))
    return records


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def save_records(records: List[RunRecord], out_dir: str) -> Tuple[str, str]:
    """Save records to JSONL and CSV.  Returns (jsonl_path, csv_path)."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(out_dir, f"l1_records_{ts}.jsonl")
    csv_path = os.path.join(out_dir, f"l1_records_{ts}.csv")

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            d = {
                "condition": r.condition,
                "constraint_label": r.constraint_label,
                "agent_id": r.agent_id,
                "agent_name": r.agent_name,
                "nationality": r.nationality,
                "faction": r.faction,
                "repeat": r.repeat,
                "stance_score": r.stance_score,
                "raw_response": r.raw_response,
                "timestamp": r.timestamp,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # CSV
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "condition", "constraint_label", "agent_id", "agent_name",
            "nationality", "faction", "repeat", "stance_score", "timestamp",
        ])
        for r in records:
            writer.writerow([
                r.condition, r.constraint_label, r.agent_id, r.agent_name,
                r.nationality, r.faction, r.repeat, r.stance_score, r.timestamp,
            ])

    return jsonl_path, csv_path


def compute_summary(records: List[RunRecord]) -> List[Dict[str, Any]]:
    """Compute per-condition × constraint × agent summary statistics."""
    import statistics
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    for r in records:
        if r.stance_score is None:
            continue
        key = (r.condition, r.constraint_label,
               r.agent_id if r.condition != "raw" else "__raw__")
        groups.setdefault(key, []).append(r.stance_score)

    summary = []
    for (cond, clabel, agent_id), scores in sorted(groups.items()):
        n = len(scores)
        mean_ = statistics.mean(scores)
        sd_ = statistics.stdev(scores) if n >= 2 else 0.0
        # 95% CI via normal approximation
        se = sd_ / (n ** 0.5) if n > 0 else 0
        ci_lo = mean_ - 1.96 * se
        ci_hi = mean_ + 1.96 * se
        # find agent info
        agent_name = ""
        nationality = ""
        faction = ""
        if cond != "raw":
            for r_ in records:
                if r_.agent_id == agent_id:
                    agent_name = r_.agent_name
                    nationality = r_.nationality
                    faction = r_.faction
                    break
        summary.append({
            "condition": cond,
            "constraint_label": clabel,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "nationality": nationality,
            "faction": faction,
            "n": n,
            "mean_stance": round(mean_, 4),
            "sd": round(sd_, 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
        })
    return summary


def save_summary(summary: List[Dict[str, Any]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"l1_summary_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def print_key_findings(summary: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary of the most important comparisons."""
    print("\n" + "=" * 70)
    print("L1 EQUIVALENT PATH – KEY FINDINGS")
    print("=" * 70)

    # 1) raw baseline across constraint levels
    print("\n--- Raw LLM baseline (no persona) ---")
    for s in summary:
        if s["condition"] == "raw":
            print(f"  {s['constraint_label']:20s}  mean={s['mean_stance']:+.3f}  "
                  f"CI=[{s['ci_95_lo']:+.3f}, {s['ci_95_hi']:+.3f}]  n={s['n']}")

    # 2) raw_role: compare each agent to raw baseline
    print("\n--- Role effect (raw_role mean - raw baseline mean at R1) ---")
    raw_r1 = next((s["mean_stance"] for s in summary
                   if s["condition"] == "raw" and s["constraint_label"] == "R1_baseline"), None)
    if raw_r1 is not None:
        for s in sorted(summary, key=lambda x: x.get("mean_stance", 0) or 0):
            if s["condition"] == "raw_role" and s["constraint_label"] == "R1_baseline":
                delta = (s["mean_stance"] or 0) - raw_r1
                print(f"  {s['agent_name']:20s} ({s['nationality']:6s} {s['faction']:8s})  "
                      f"mean={s['mean_stance']:+.3f}  Δ={delta:+.3f}  "
                      f"CI=[{s['ci_95_lo']:+.3f}, {s['ci_95_hi']:+.3f}]")

    # 3) constraint effect: how does adding constraints shift the raw stance?
    print("\n--- Constraint effect on raw LLM (R1 → R18 shift) ---")
    raw_r18 = next((s["mean_stance"] for s in summary
                    if s["condition"] == "raw" and s["constraint_label"] == "R18_full"), None)
    if raw_r1 is not None and raw_r18 is not None:
        print(f"  Raw R1: {raw_r1:+.3f}  →  Raw R18: {raw_r18:+.3f}  "
              f"(shift = {raw_r18 - raw_r1:+.3f})")

    # 4) Chinese vs Foreign role stance difference
    print("\n--- Chinese vs Foreign agent stances (raw_role, R1) ---")
    cn_scores = []
    fn_scores = []
    for s in summary:
        if s["condition"] == "raw_role" and s["constraint_label"] == "R1_baseline":
            if s["nationality"] == "中国":
                cn_scores.append(s["mean_stance"] or 0)
            else:
                fn_scores.append(s["mean_stance"] or 0)
    if cn_scores and fn_scores:
        import statistics
        print(f"  Chinese agents:  mean={statistics.mean(cn_scores):+.3f}  n={len(cn_scores)}")
        print(f"  Foreign agents:  mean={statistics.mean(fn_scores):+.3f}  n={len(fn_scores)}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L1: Equivalent Path Baseline – raw LLM vs role-prompted stance"
    )
    p.add_argument("--raw-repeats", type=int, default=20,
                   help="Repeats for raw (no-persona) baseline")
    p.add_argument("--role-repeats", type=int, default=5,
                   help="Repeats per role per constraint level")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override LLM temperature (default: use config value)")
    p.add_argument("--provider", type=str, default="",
                   help="LLM provider override (ecnu|gitee|gitee-qwen|...)")
    p.add_argument("--model", type=str, default="",
                   help="Explicit model name override")
    p.add_argument("--constraints", type=str,
                   default="R1_baseline,R10_transition,R18_full",
                   help="Comma-separated constraint labels to test")
    p.add_argument("--out-dir", type=str, default=OUT_DIR,
                   help="Output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    provider = args.provider or cfg.model_provider or "ecnu"
    print(f"L1 Equivalent Path Baseline")
    print(f"  provider={provider}  model={args.model or cfg.ecnu_model}")
    print(f"  raw_repeats={args.raw_repeats}  role_repeats={args.role_repeats}")
    print(f"  constraints={args.constraints}")

    llm = _build_llm(cfg, provider, explicit_model=args.model)
    if args.temperature is not None:
        llm.temperature = args.temperature

    constraint_labels = [c.strip() for c in args.constraints.split(",") if c.strip()]
    roles = [r for r in BASE_ROLES]  # 11 roles

    all_records: List[RunRecord] = []

    # --- raw baseline ---
    print("\n[1/2] Running raw (no-persona) baseline ...")
    raw_records = run_raw_baseline(llm, args.temperature or 0.7,
                                   args.raw_repeats, constraint_labels)
    all_records.extend(raw_records)
    print(f"  -> {len(raw_records)} records")

    # --- role baseline ---
    print("\n[2/2] Running role-prompted baseline ...")
    role_records = run_role_baseline(llm, roles, args.role_repeats, constraint_labels)
    all_records.extend(role_records)
    print(f"  -> {len(role_records)} records")

    # --- save ---
    jsonl_path, csv_path = save_records(all_records, args.out_dir)
    print(f"\nRecords saved:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")

    # --- summary ---
    summary = compute_summary(all_records)
    summary_path = save_summary(summary, args.out_dir)
    print(f"  Summary: {summary_path}")

    # --- print findings ---
    print_key_findings(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
