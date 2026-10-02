import csv
import hashlib
import json
import math
import os
import re
import time
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.runtime_store import runtime_store
from utils.stance_eval import evaluate_agent_stance
from utils.text_fix import repair_text

try:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)


class GraphPlotter:
    STANCE_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    STANCE_LABELS = [
        "-1 强烈反对多党轮流执政",
        "-0.5 倾向反对",
        "0 中立/观望",
        "+0.5 倾向支持",
        "+1 强烈支持多党轮流执政",
    ]
    def __init__(self, eval_llm=None, llm=None, logger: Optional[Callable[[str], None]] = None):
        self.eval_llm = eval_llm
        self.llm = llm
        self._logger = logger or (lambda _: None)

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        except Exception:
            pass

    def resolve_model_name(self, model_info: Dict[str, Any], fallback_llm: Any = None) -> str:
        if isinstance(model_info, dict):
            for key in ["model", "eval_model", "model_name", "base_model", "name"]:
                v = model_info.get(key)
                if isinstance(v, str) and v.strip() and v.strip().lower() != "unknown":
                    return v.strip()
            provider = (model_info.get("provider") or model_info.get("eval_provider") or "").strip()
            model = (model_info.get("model") or model_info.get("eval_model") or "").strip()
            if provider and model and model.lower() != "unknown":
                return f"{provider}-{model}"
        llm_obj = fallback_llm or self.llm
        for key in ["model", "model_name", "name"]:
            v = getattr(llm_obj, key, None)
            if isinstance(v, str) and v.strip() and v.strip().lower() != "unknown":
                return v.strip()
        provider = getattr(llm_obj, "provider", None)
        if isinstance(provider, str) and provider.strip():
            return f"{provider.strip()}-model"
        return "model_missing"

    @staticmethod
    def _safe_name(value: str) -> str:
        txt = str(value or "")
        return "".join([c if c.isalnum() or c in "-_." else "_" for c in txt])

    @staticmethod
    def _stance_score(ev: Dict[str, Any]) -> Optional[float]:
        if not isinstance(ev, dict):
            return None
        try:
            return float(ev.get("support", 0.0)) - float(ev.get("oppose", 0.0))
        except Exception:
            return None

    def _infer_stance_from_round(self, rlog: Dict[str, Any], agent: str, language_mode: str) -> Optional[float]:
        interactions = rlog.get("interactions", []) or []
        for it in interactions:
            if not isinstance(it, dict) or it.get("agent") != agent:
                continue
            text = repair_text((str(it.get("message") or "") + " " + str(it.get("stance") or "")).strip())
            if not text:
                continue
            ev = evaluate_agent_stance(self.eval_llm, agent, text, language_mode=language_mode)
            return self._stance_score(ev)
        return None

    @staticmethod
    def _series_from_rounds(round_logs_sorted: List[Dict[str, Any]], agent_names: List[str], infer_fn) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {a: [] for a in agent_names}
        last_vals: Dict[str, float] = {a: 0.0 for a in agent_names}
        for rlog in round_logs_sorted:
            se = rlog.get("stance_eval") or {}
            for a in agent_names:
                val = None
                if isinstance(se, dict):
                    val = GraphPlotter._stance_score(se.get(a))
                if val is None:
                    val = infer_fn(rlog, a)
                if val is not None:
                    last_vals[a] = float(val)
                out[a].append(float(last_vals[a]))
        for a in agent_names:
            if not out[a]:
                out[a] = [0.0]
        return out

    @staticmethod
    def _pressure_by_round(round_logs_sorted: List[Dict[str, Any]], agent_count: int) -> List[float]:
        pressures = []
        # determine total number of distinct feature/constraint entries actually used in this simulation
        total_constraints_set = set()
        for r in round_logs_sorted:
            for c in (r.get("event_constraints") or []) or []:
                try:
                    total_constraints_set.add(str(c))
                except Exception:
                    continue
        total_constraints_count = max(6, len(total_constraints_set))
        for rlog in round_logs_sorted:
            inductions = rlog.get("inductions") or {}
            total_induction = 0.0
            if isinstance(inductions, dict):
                for _, items in inductions.items():
                    if not isinstance(items, list):
                        continue
                    for it in items:
                        if isinstance(it, dict):
                            total_induction += float(it.get("intensity", it.get("score", 0)) or 0)
            constraints = rlog.get("event_constraints") or []
            # use the number of feature entries actually present in the simulation (not a hardcoded 6)
            # keep c_weight as at least 1.0 to avoid division by zero
            c_weight = max(1.0, float(total_constraints_count))
            avg_induction = total_induction / max(1, agent_count)
            pressures.append(float(avg_induction / c_weight))
        return pressures

    def _save_io_exports(self, base: str) -> None:
        traces = runtime_store.agent_traces or {}
        jpath = os.path.join(PLOTS_DIR, f"{base}_agent_traces.json")
        cpath = os.path.join(PLOTS_DIR, f"{base}_agent_traces.csv")
        with open(jpath, "w", encoding="utf-8") as jf:
            jf.write(json.dumps(traces, ensure_ascii=False, indent=2))
        with open(cpath, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["agent", "step", "round", "prompt", "output", "meta"])
            for agent, entries in traces.items():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    meta = e.get("meta", {}) if isinstance(e, dict) else {}
                    writer.writerow([
                        agent,
                        (e.get("step") if isinstance(e, dict) else "") or "",
                        meta.get("round") or meta.get("round_idx") or "",
                        (e.get("prompt") if isinstance(e, dict) else "") or "",
                        (e.get("output") if isinstance(e, dict) else "") or "",
                        json.dumps(meta, ensure_ascii=False),
                    ])

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score <= -0.6:
            return "strong_oppose"
        if score <= -0.2:
            return "oppose"
        if score < 0.2:
            return "neutral"
        if score < 0.6:
            return "support"
        return "strong_support"

    def _save_induction_feature_stats(self, round_logs_sorted: List[Dict[str, Any]], agent_names: List[str], agent_series: Dict[str, List[float]], base: str) -> None:
        stats_rows: List[Dict[str, Any]] = []
        per_round_total: Dict[int, Dict[str, int]] = {}
        for r_idx, rlog in enumerate(round_logs_sorted):
            r = int(rlog.get("round") or 0)
            inductions = rlog.get("inductions") or {}
            influences = rlog.get("influences") or []
            interactions = rlog.get("interactions") or []
            interaction_by_agent: Dict[str, Dict[str, Any]] = {}
            for it in interactions:
                if isinstance(it, dict) and it.get("agent"):
                    interaction_by_agent[str(it.get("agent"))] = it
            influence_by_agent: Dict[str, Dict[str, Any]] = {}
            for inf in influences:
                if isinstance(inf, dict) and inf.get("agent"):
                    influence_by_agent[str(inf.get("agent"))] = inf
            for a in agent_names:
                recv_items = (inductions.get(a) or []) if isinstance(inductions, dict) else []
                recv_ids = []
                recv_content_to_id: Dict[str, str] = {}
                for it in recv_items:
                    if isinstance(it, dict):
                        iid = str(it.get("id") or "").strip()
                        content = str(it.get("content") or "").strip()
                        recv_ids.append(iid or content or "")
                        if iid and content:
                            recv_content_to_id[content.lower()] = iid
                inf = influence_by_agent.get(a, {}) if isinstance(influence_by_agent, dict) else {}
                adopted_raw = inf.get("from_inductions", []) if isinstance(inf, dict) else []
                adopted_ids: List[str] = []
                for x in (adopted_raw or []):
                    sx = str(x or "").strip()
                    if not sx:
                        continue
                    # Case 1: already an id
                    if sx in recv_ids:
                        adopted_ids.append(sx)
                        continue
                    # Case 2: content text; map back to id
                    mapped = recv_content_to_id.get(sx.lower())
                    if mapped:
                        adopted_ids.append(mapped)
                        continue
                    # Case 3: partial content match
                    best_id = ""
                    best_len = 0
                    for content_low, iid in recv_content_to_id.items():
                        if sx.lower() in content_low or content_low in sx.lower():
                            mlen = min(len(sx), len(content_low))
                            if mlen > best_len:
                                best_len = mlen
                                best_id = iid
                    adopted_ids.append(best_id or sx)
                recv_set = set([x for x in recv_ids if x])
                adopted_set = set(adopted_ids)
                adopted_hit = sorted(list(recv_set.intersection(adopted_set)))
                # objective adoption estimate from generated message (not relying on self-report)
                msg = ""
                it = interaction_by_agent.get(a) or {}
                if isinstance(it, dict):
                    parsed = it.get("parsed") if isinstance(it.get("parsed"), dict) else {}
                    msg = str(it.get("message") or "") + " " + str(parsed.get("message") or "")
                msg_low = msg.lower()
                adopted_judged_ids: List[str] = []
                for item in recv_items:
                    if not isinstance(item, dict):
                        continue
                    iid = str(item.get("id") or "").strip()
                    content = str(item.get("content") or "")
                    if not iid:
                        continue
                    tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{3,}", content.lower())
                    if not tokens:
                        continue
                    hit = sum(1 for tk in tokens[:8] if tk and tk in msg_low)
                    if hit >= 1:
                        adopted_judged_ids.append(iid)
                score_now = float((agent_series.get(a) or [0.0])[r_idx]) if (agent_series.get(a) and len(agent_series.get(a)) > r_idx) else 0.0
                score_prev = float((agent_series.get(a) or [score_now])[r_idx - 1]) if r_idx > 0 and (agent_series.get(a) and len(agent_series.get(a)) > r_idx - 1) else score_now
                delta = score_now - score_prev
                stance_changed_judged = abs(delta) >= 0.2 or (self._label_from_score(score_prev) != self._label_from_score(score_now))
                support_to_oppose_judged = score_prev >= 0.2 and score_now <= -0.2
                row = {
                    "round": r,
                    "agent": a,
                    "received_count": len([x for x in recv_ids if x]),
                    "received_ids": recv_ids,
                    "adopted_declared_count": len(adopted_ids),
                    "adopted_declared_ids": adopted_ids,
                    "adopted_hit_count": len(adopted_hit),
                    "adopted_hit_ids": adopted_hit,
                    "adopted_judged_count": len(adopted_judged_ids),
                    "adopted_judged_ids": sorted(set(adopted_judged_ids)),
                    "changed": bool((inf or {}).get("changed", False)),
                    "stance_changed_judged": stance_changed_judged,
                    "support_to_oppose_judged": support_to_oppose_judged,
                    "stance_prev_score": score_prev,
                    "stance_now_score": score_now,
                    "stance_delta": delta,
                    "stance_prev_label": self._label_from_score(score_prev),
                    "stance_now_label": self._label_from_score(score_now),
                }
                stats_rows.append(row)
                bucket = per_round_total.setdefault(r, {"received": 0, "adopted": 0, "changed": 0, "adopted_judged": 0, "changed_judged": 0, "support_to_oppose_judged": 0})
                bucket["received"] += row["received_count"]
                bucket["adopted"] += row["adopted_hit_count"]
                bucket["changed"] += int(row["changed"])
                bucket["adopted_judged"] += int(row["adopted_judged_count"])
                bucket["changed_judged"] += int(bool(row["stance_changed_judged"]))
                bucket["support_to_oppose_judged"] += int(bool(row["support_to_oppose_judged"]))

        injected_rounds = [
            int(r.get("round") or 0)
            for r in round_logs_sorted
            if bool(r.get("event_constraints_changed"))
            or (int(r.get("round") or 0) == 1 and bool(r.get("event_constraints")))
        ]
        feature_effects: List[Dict[str, Any]] = []
        for rr in sorted(set(injected_rounds)):
            ridx = next((i for i, rlog in enumerate(round_logs_sorted) if int(rlog.get("round") or 0) == rr), None)
            if ridx is None:
                continue
            for a in agent_names:
                vals = agent_series.get(a) or []
                if ridx >= len(vals):
                    continue
                prev = float(vals[ridx - 1]) if ridx > 0 else float(vals[ridx])
                now = float(vals[ridx])
                flip_support_to_oppose = prev >= 0.2 and now <= -0.2
                significant = flip_support_to_oppose or abs(now - prev) >= 0.6
                feature_effects.append({
                    "round": rr,
                    "agent": a,
                    "prev_score": prev,
                    "now_score": now,
                    "delta": now - prev,
                    "prev_label": self._label_from_score(prev),
                    "now_label": self._label_from_score(now),
                    "significant_change": significant,
                    "support_to_oppose": flip_support_to_oppose,
                })

        summary = {
            "round_count": len(round_logs_sorted),
            "agent_count": len(agent_names),
            "injected_rounds": sorted(set(injected_rounds)),
            "total_received": sum(v["received"] for v in per_round_total.values()),
            "total_adopted_hit": sum(v["adopted"] for v in per_round_total.values()),
            "total_changed": sum(v["changed"] for v in per_round_total.values()),
            "total_adopted_judged": sum(int(x.get("adopted_judged_count", 0)) for x in stats_rows),
            "total_stance_changed_judged": sum(int(bool(x.get("stance_changed_judged"))) for x in stats_rows),
            "support_to_oppose_cases": [x for x in feature_effects if x.get("support_to_oppose")],
            "significant_feature_effect_cases": [x for x in feature_effects if x.get("significant_change")],
        }
        # induction_id -> agent adoption count summary
        id_stats: Dict[str, Dict[str, Any]] = {}
        for row in stats_rows:
            agent = str(row.get("agent") or "")
            rr = int(row.get("round") or 0)
            for iid in row.get("adopted_declared_ids", []) or []:
                key = str(iid or "").strip()
                if not key:
                    continue
                bucket = id_stats.setdefault(key, {"induction_id": key, "adopted_total_count": 0, "adopted_by_agents": set(), "adopted_count_by_agent": {}, "rounds_used": set(), "adoption_source": {"declared": 0, "judged": 0}})
                bucket["adopted_total_count"] += 1
                bucket["adopted_by_agents"].add(agent)
                bucket["adopted_count_by_agent"][agent] = int(bucket["adopted_count_by_agent"].get(agent, 0)) + 1
                bucket["rounds_used"].add(rr)
                bucket["adoption_source"]["declared"] += 1
            for iid in row.get("adopted_judged_ids", []) or []:
                key = str(iid or "").strip()
                if not key:
                    continue
                bucket = id_stats.setdefault(key, {"induction_id": key, "adopted_total_count": 0, "adopted_by_agents": set(), "adopted_count_by_agent": {}, "rounds_used": set(), "adoption_source": {"declared": 0, "judged": 0}})
                bucket["adoption_source"]["judged"] += 1
        id_stats_rows = []
        for _, v in id_stats.items():
            id_stats_rows.append({
                "induction_id": v["induction_id"],
                "adopted_total_count": int(v["adopted_total_count"]),
                "adopted_by_agents": sorted(list(v["adopted_by_agents"])),
                "adopted_count_by_agent": v["adopted_count_by_agent"],
                "rounds_used": sorted(list(v["rounds_used"])),
                "adoption_source": v["adoption_source"],
            })
        id_stats_rows = sorted(id_stats_rows, key=lambda x: (-int(x.get("adopted_total_count", 0)), str(x.get("induction_id") or "")))
        checklist = {
            "by_round": {},
        }
        for row in stats_rows:
            rr = str(row["round"])
            checklist["by_round"].setdefault(rr, {})
            checklist["by_round"][rr][row["agent"]] = {
                "received_induction_ids": row["received_ids"],
                "adopted_induction_ids_declared": row["adopted_declared_ids"],
                "adopted_induction_ids_judged": row["adopted_judged_ids"],
                "stance_prev_label": row["stance_prev_label"],
                "stance_now_label": row["stance_now_label"],
                "support_to_oppose_judged": row["support_to_oppose_judged"],
                "stance_changed_judged": row["stance_changed_judged"],
            }
        out_json = os.path.join(PLOTS_DIR, f"{base}_induction_feature_stats.json")
        out_checklist_json = os.path.join(PLOTS_DIR, f"{base}_induction_checklist_by_round.json")
        out_csv = os.path.join(PLOTS_DIR, f"{base}_induction_feature_stats.csv")
        out_judged_json = os.path.join(PLOTS_DIR, f"{base}_induction_judged_summary.json")
        out_judged_csv = os.path.join(PLOTS_DIR, f"{base}_induction_judged_summary.csv")
        out_id_json = os.path.join(PLOTS_DIR, f"{base}_induction_id_adoption_stats.json")
        out_id_csv = os.path.join(PLOTS_DIR, f"{base}_induction_id_adoption_stats.csv")
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "summary": summary,
                    "per_round_total": per_round_total,
                    "per_agent_round_stats": stats_rows,
                    "feature_effects": feature_effects,
                    "induction_id_adoption_stats": id_stats_rows,
                },
                jf,
                ensure_ascii=False,
                indent=2,
            )
        with open(out_checklist_json, "w", encoding="utf-8") as jf:
            json.dump(checklist, jf, ensure_ascii=False, indent=2)
        with open(out_judged_json, "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "summary": {
                        "total_adopted_judged": summary.get("total_adopted_judged", 0),
                        "total_stance_changed_judged": summary.get("total_stance_changed_judged", 0),
                    },
                    "per_round_judged": {
                        str(r): {
                            "adopted_judged": v.get("adopted_judged", 0),
                            "changed_judged": v.get("changed_judged", 0),
                            "support_to_oppose_judged": v.get("support_to_oppose_judged", 0),
                        }
                        for r, v in per_round_total.items()
                    },
                    "per_agent_round_judged": [
                        {
                            "round": row["round"],
                            "agent": row["agent"],
                            "adopted_induction_ids_judged": row["adopted_judged_ids"],
                            "adopted_judged_count": row["adopted_judged_count"],
                            "stance_changed_judged": row["stance_changed_judged"],
                            "support_to_oppose_judged": row["support_to_oppose_judged"],
                            "stance_prev_label": row["stance_prev_label"],
                            "stance_now_label": row["stance_now_label"],
                        }
                        for row in stats_rows
                    ],
                },
                jf,
                ensure_ascii=False,
                indent=2,
            )
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "round", "agent", "received_count", "adopted_declared_count", "adopted_hit_count",
                "adopted_judged_count",
                "changed", "stance_prev_score", "stance_now_score", "stance_delta",
                "stance_prev_label", "stance_now_label", "stance_changed_judged", "support_to_oppose_judged",
            ])
            for row in stats_rows:
                writer.writerow([
                    row["round"], row["agent"], row["received_count"], row["adopted_declared_count"], row["adopted_hit_count"],
                    row["adopted_judged_count"],
                    row["changed"], row["stance_prev_score"], row["stance_now_score"], row["stance_delta"],
                    row["stance_prev_label"], row["stance_now_label"], row["stance_changed_judged"], row["support_to_oppose_judged"],
                ])
        with open(out_judged_csv, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["round", "agent", "adopted_judged_count", "stance_changed_judged", "support_to_oppose_judged", "stance_prev_label", "stance_now_label"])
            for row in stats_rows:
                writer.writerow([
                    row["round"], row["agent"], row["adopted_judged_count"], row["stance_changed_judged"],
                    row["support_to_oppose_judged"], row["stance_prev_label"], row["stance_now_label"],
                ])
        with open(out_id_json, "w", encoding="utf-8") as jf:
            json.dump({"induction_id_adoption_stats": id_stats_rows}, jf, ensure_ascii=False, indent=2)
        with open(out_id_csv, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["induction_id", "adopted_total_count", "adopted_by_agents", "adopted_count_by_agent", "rounds_used", "declared_count", "judged_count"])
            for row in id_stats_rows:
                src = row.get("adoption_source", {}) or {}
                writer.writerow([
                    row.get("induction_id", ""),
                    row.get("adopted_total_count", 0),
                    "|".join(row.get("adopted_by_agents", []) or []),
                    json.dumps(row.get("adopted_count_by_agent", {}), ensure_ascii=False),
                    "|".join([str(x) for x in (row.get("rounds_used", []) or [])]),
                    int(src.get("declared", 0)),
                    int(src.get("judged", 0)),
                ])
        self._log(f"[induction_analysis] saved summary: {out_json}")
        explain_path = os.path.join(PLOTS_DIR, f"{base}_induction_stats_explain.txt")
        with open(explain_path, "w", encoding="utf-8") as f:
            f.write(
                "图表说明:\n"
                "1. 收到诱导(浅蓝柱): 本轮所有agent收到的诱导条目总数。\n"
                "2. 采用诱导(深蓝柱): 收到ID与agent自报from_inductions的交集总数。\n"
                "3. 立场变化人数(红线): 本轮changed=true的agent数量。\n"
                "4. 紫色虚线: 特征集注入轮(event_constraints_changed=true)。\n"
                "5. 非自述判定: *_induction_judged_summary.json/csv 为独立判定产物；并在图中用虚线展示。\n"
            )

        try:
            import matplotlib.pyplot as plt  # type: ignore
            rounds = sorted(per_round_total.keys())
            recv = [per_round_total[r]["received"] for r in rounds]
            adopt = [per_round_total[r]["adopted"] for r in rounds]
            changed = [per_round_total[r]["changed"] for r in rounds]
            adopt_j = [per_round_total[r]["adopted_judged"] for r in rounds]
            changed_j = [per_round_total[r]["changed_judged"] for r in rounds]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(rounds, recv, color="#93c5fd", label="收到诱导条目数")
            ax.bar(rounds, adopt, color="#2563eb", label="采用诱导条目数(交集)")
            ax.plot(rounds, changed, color="#dc2626", marker="o", linewidth=1.5, label="立场变化人数(自报)")
            ax.plot(rounds, adopt_j, color="#0f766e", marker="x", linewidth=1.2, linestyle="--", label="采用人数(非自述判定)")
            ax.plot(rounds, changed_j, color="#7c2d12", marker="s", linewidth=1.2, linestyle="--", label="立场变化人数(非自述判定)")
            for rr in sorted(set(injected_rounds)):
                ax.axvline(rr, linestyle="--", linewidth=1.0, color="#7c3aed", alpha=0.55)
            ax.set_xlabel("轮次")
            ax.set_ylabel("数量")
            ax.set_title("每轮诱导接收与采用统计")
            ax.legend(fontsize=8)
            out_chart = os.path.join(PLOTS_DIR, f"{base}_induction_stats.jpg")
            fig.tight_layout()
            fig.savefig(out_chart, dpi=150, format="jpg")
            plt.close(fig)
        except Exception:
            pass

    def _parse_agent_profiles_from_traces(self, agent_names: List[str]) -> List[Dict[str, Any]]:
        traces = runtime_store.agent_traces or {}
        rows: List[Dict[str, Any]] = []
        for agent in agent_names:
            profile = {
                "agent": agent,
                "nationality": "",
                "political_identity": "",
                "faction": "",
                "regime_stance": "",
                "authority_preference": "",
                "legitimacy_source": "",
                "distribution_preference": "",
                "belief": "",
                "income_level": "",
                "occupation_type": "",
                "work_intensity": "",
                "has_children": "",
                "education_anxiety": "",
                "family_burden": "",
            }
            entries = traces.get(agent, []) if isinstance(traces.get(agent, []), list) else []
            init_prompt = ""
            for e in entries:
                if isinstance(e, dict) and str(e.get("step") or "") in ["init_stance", "decide"]:
                    txt = str(e.get("prompt") or "")
                    if txt:
                        init_prompt = txt
                        break
            if not init_prompt:
                rows.append(profile)
                continue
            pairs = [
                (r"国籍[：:]\s*([^，。\n]+)", "nationality"),
                (r"政治身份[：:]\s*([^，。\n]+)", "political_identity"),
                (r"阵营[：:]\s*([^，。\n]+)", "faction"),
                (r"政体立场[：:]\s*([^；。\n]+)", "regime_stance"),
                (r"权力偏好[：:]\s*([^；。\n]+)", "authority_preference"),
                (r"合法性来源[：:]\s*([^；。\n]+)", "legitimacy_source"),
                (r"分配倾向[：:]\s*([^；。\n]+)", "distribution_preference"),
                (r"信念[：:]\s*([^；。\n]+)", "belief"),
                (r"收入([高中低]|中)", "income_level"),
                (r"职业类型([a-zA-Z_]+)", "occupation_type"),
                (r"工作强度([高中低]|中)", "work_intensity"),
                (r"有子女|无子女", "has_children"),
                (r"教育焦虑([0-9.]+)", "education_anxiety"),
                (r"家庭负担([0-9.]+)", "family_burden"),
            ]
            for pat, key in pairs:
                m = re.search(pat, init_prompt)
                if m:
                    if key in ["has_children"]:
                        profile[key] = m.group(0)
                    elif m.groups():
                        profile[key] = m.group(1).strip()
                    else:
                        profile[key] = m.group(0).strip()
            rows.append(profile)
        return rows

    @staticmethod
    def _wrap_cell_text(v: Any, width: int = 16) -> str:
        s = str(v or "")
        if not s:
            return ""
        return "\n".join(textwrap.wrap(s, width=width, break_long_words=True, replace_whitespace=False))

    def _save_agent_profile_table(self, agent_names: List[str], base: str) -> None:
        rows = self._parse_agent_profiles_from_traces(agent_names)
        jpath = os.path.join(PLOTS_DIR, f"{base}_agent_profiles.json")
        cpath = os.path.join(PLOTS_DIR, f"{base}_agent_profiles.csv")
        with open(jpath, "w", encoding="utf-8") as jf:
            json.dump({"agent_profiles": rows}, jf, ensure_ascii=False, indent=2)
        headers = [
            "agent", "nationality", "political_identity", "faction", "regime_stance",
            "authority_preference", "legitimacy_source", "distribution_preference", "belief",
            "income_level", "occupation_type", "work_intensity", "has_children",
            "education_anxiety", "family_burden",
        ]
        with open(cpath, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(headers)
            for r in rows:
                writer.writerow([r.get(h, "") for h in headers])
        # also produce an easy-to-read image table
        try:
            import matplotlib.pyplot as plt  # type: ignore
            fig, ax = plt.subplots(figsize=(13, max(4, len(rows) * 0.55)))
            ax.axis("off")
            show_cols = ["agent", "nationality", "political_identity", "regime_stance", "income_level", "education_anxiety", "family_burden"]
            table_data = [[self._wrap_cell_text(r.get(c, ""), width=16) for c in show_cols] for r in rows]
            tb = ax.table(cellText=table_data, colLabels=show_cols, loc="center")
            tb.auto_set_font_size(False)
            tb.set_fontsize(8)
            # Increase row height to avoid overlap after wrapping.
            tb.scale(1, 1.9)
            ax.set_title("Agent 设定总览表")
            out = os.path.join(PLOTS_DIR, f"{base}_agent_profiles_table.jpg")
            fig.tight_layout()
            fig.savefig(out, dpi=150, format="jpg")
            plt.close(fig)
        except Exception:
            pass

    def _maybe_save_agent_profile_table(self, agent_names: List[str], base: str) -> None:
        # Default behavior: do not export every run.
        force_export = bool((runtime_store.state or {}).get("force_agent_profile_export", False))
        always_export = bool((runtime_store.state or {}).get("export_agent_profiles_each_run", False))
        signature = "|".join(sorted([str(x) for x in (agent_names or [])]))
        sig_hash = hashlib.md5(signature.encode("utf-8", errors="ignore")).hexdigest()
        meta_path = os.path.join(PLOTS_DIR, "agent_profiles_latest.meta.json")
        prev_hash = ""
        if os.path.exists(meta_path):
            try:
                prev_hash = str(json.loads(open(meta_path, "r", encoding="utf-8").read()).get("agent_hash") or "")
            except Exception:
                prev_hash = ""
        if not (force_export or always_export) and prev_hash == sig_hash:
            return
        self._save_agent_profile_table(agent_names, base)
        latest_json = os.path.join(PLOTS_DIR, "agent_profiles_latest.json")
        latest_csv = os.path.join(PLOTS_DIR, "agent_profiles_latest.csv")
        try:
            rows = self._parse_agent_profiles_from_traces(agent_names)
            with open(latest_json, "w", encoding="utf-8") as jf:
                json.dump({"agent_profiles": rows}, jf, ensure_ascii=False, indent=2)
            headers = [
                "agent", "nationality", "political_identity", "faction", "regime_stance",
                "authority_preference", "legitimacy_source", "distribution_preference", "belief",
                "income_level", "occupation_type", "work_intensity", "has_children",
                "education_anxiety", "family_burden",
            ]
            with open(latest_csv, "w", encoding="utf-8-sig", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(headers)
                for r in rows:
                    writer.writerow([r.get(h, "") for h in headers])
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({"agent_hash": sig_hash, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, mf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _plot_agent_stance_matrix(self, round_logs_sorted: List[Dict[str, Any]], agent_names: List[str], agent_series: Dict[str, List[float]], base: str) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return
        if not round_logs_sorted or not agent_names:
            return
        rounds = [int(r.get("round") or 0) for r in round_logs_sorted]
        matrix = []
        for a in agent_names:
            vals = agent_series.get(a) or []
            if len(vals) < len(rounds):
                vals = vals + [vals[-1] if vals else 0.0] * (len(rounds) - len(vals))
            matrix.append(vals[: len(rounds)])
        arr = np.array(matrix, dtype=float) if matrix else np.zeros((0, 0))
        fig, ax = plt.subplots(figsize=(10, max(4, len(agent_names) * 0.45)))
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_yticks(range(len(agent_names)))
        ax.set_yticklabels(agent_names)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([f"R{r}" for r in rounds])
        ax.set_title("全体Agent逐轮立场矩阵")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("立场分值")
        cbar.set_ticks(self.STANCE_TICKS)
        cbar.set_ticklabels(self.STANCE_LABELS)
        out = os.path.join(PLOTS_DIR, f"{base}_agent_stance_matrix.jpg")
        fig.tight_layout()
        fig.savefig(out, dpi=150, format="jpg")
        plt.close(fig)

    @staticmethod
    def _hash_embed(text: str, dim: int = 128) -> List[float]:
        vec = [0.0] * dim
        if not text:
            return vec
        for token in text.split():
            h = int(hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest(), 16)
            idx = h % dim
            sign = -1.0 if (h >> 1) % 2 else 1.0
            vec[idx] += sign * (1.0 + (len(token) % 3) * 0.1)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = model.encode(texts, normalize_embeddings=True)
            return [list(map(float, row)) for row in embs]
        except Exception:
            return [self._hash_embed(t) for t in texts]

    @staticmethod
    def _pca_2d(vectors: List[List[float]]) -> List[Tuple[float, float]]:
        if not vectors:
            return []
        try:
            import numpy as np  # type: ignore
            arr = np.array(vectors, dtype=float)
            arr = arr - arr.mean(axis=0, keepdims=True)
            u, s, _ = np.linalg.svd(arr, full_matrices=False)
            z = u[:, :2] * s[:2]
            return [(float(x), float(y)) for x, y in z]
        except Exception:
            return [(float(i), 0.0) for i in range(len(vectors))]

    def _logical_consistency(self, statement: str, constraints: List[str]) -> float:
        if not statement:
            return 0.3
        prompt = (
            "Evaluate logical consistency from 0 to 1. Output only a number. "
            f"Constraints: {constraints}\n"
            f"Statement: {statement}\n"
        )
        if self.eval_llm is not None:
            try:
                raw = self.eval_llm.generate(prompt)
                val = float(str(raw).strip().split()[0])
                return max(0.0, min(1.0, val))
            except Exception:
                pass
        has_reason = ("because" in statement.lower()) or ("因此" in statement) or ("论据" in statement)
        return 0.7 if has_reason else 0.45

    def _plot_radar(self, round_logs_sorted: List[Dict[str, Any]], agent_names: List[str], base: str) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return
        if not round_logs_sorted or not agent_names:
            return
        latest = round_logs_sorted[-1]
        # build the set of distinct constraints that actually appeared in the simulated rounds
        constraint_set = set()
        for r in round_logs_sorted:
            for c in (r.get("event_constraints") or []) or []:
                try:
                    constraint_set.add(str(c))
                except Exception:
                    continue
        constraints = [str(x) for x in (latest.get("event_constraints") or [])]
        labels = ["Induction", "Constraint", "SelfInterest", "Consistency"]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        for idx, agent in enumerate(agent_names[:8]):
            induction_strength = 0.0
            for rlog in round_logs_sorted:
                items = (rlog.get("inductions") or {}).get(agent, [])
                if isinstance(items, list):
                    induction_strength += sum(float((it or {}).get("intensity", (it or {}).get("score", 0)) or 0) for it in items if isinstance(it, dict))
            induction_strength = max(0.0, min(1.0, induction_strength / max(1.0, 6.0 * len(round_logs_sorted))))
            # normalize constraint pressure by the actual number of feature entries used in this simulation
            total_features = max(1, len(constraint_set))
            constraint_pressure = max(0.0, min(1.0, float(len(constraints)) / float(total_features)))
            self_interest = 0.5
            text = ""
            for rlog in reversed(round_logs_sorted):
                for it in rlog.get("interactions", []) or []:
                    if isinstance(it, dict) and it.get("agent") == agent:
                        text = str(it.get("message") or "")
                        break
                if text:
                    break
            if any(k in text for k in ["income", "salary", "工资", "收入", "加班费"]):
                self_interest = 0.85
            elif any(k in text for k in ["stability", "秩序", "稳定", "安全"]):
                self_interest = 0.65
            consistency = self._logical_consistency(text, constraints)
            vals = [induction_strength, constraint_pressure, self_interest, consistency]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.6, label=agent)
            ax.fill(angles, vals, alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)
        ax.set_title("Constraint-Induction Balance Radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)
        out_path = os.path.join(PLOTS_DIR, f"{base}_constraint_induction_radar.jpg")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, format="jpg")
        plt.close(fig)
        self._log(f"[stance_eval] saved radar: {out_path}")

    def _plot_potential_field(self, round_logs_sorted: List[Dict[str, Any]], agent_names: List[str], base: str) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib import patches
        except Exception:
            return
        if not round_logs_sorted or not agent_names:
            return
        target_text = "对中国来说，多党轮流执政是否是中国政体最优解"
        constraint_texts = []
        for rlog in round_logs_sorted:
            for c in rlog.get("event_constraints", []) or []:
                constraint_texts.append(str(c))
        constraint_texts = constraint_texts[:12]
        point_meta: List[Tuple[str, int, str]] = []
        point_texts: List[str] = []
        for rlog in round_logs_sorted:
            rnd = int(rlog.get("round") or 0)
            by_agent = {it.get("agent"): it for it in (rlog.get("interactions") or []) if isinstance(it, dict)}
            for a in agent_names:
                msg = str((by_agent.get(a) or {}).get("message") or "")
                stance = str((by_agent.get(a) or {}).get("stance") or "")
                point_meta.append((a, rnd, msg))
                point_texts.append((msg + " " + stance).strip())
        all_texts = [target_text] + constraint_texts + point_texts
        embs = self._embed_texts(all_texts)
        coords = self._pca_2d(embs)
        if not coords:
            return
        target_xy = coords[0]

        def shift(xy):
            return (xy[0] - target_xy[0], xy[1] - target_xy[1])

        offset = 1
        constraint_coords = [shift(c) for c in coords[offset : offset + len(constraint_texts)]]
        point_coords = [shift(c) for c in coords[offset + len(constraint_texts) :]]
        if constraint_coords:
            dists = [math.sqrt(x * x + y * y) for x, y in constraint_coords]
            r_crit = max(0.15, sum(dists) / len(dists))
        else:
            r_crit = 0.5

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.add_patch(patches.Circle((0, 0), r_crit, color="#ff6b6b", alpha=0.12, label="约束红线区"))
        ax.add_patch(patches.Circle((0, 0), r_crit * 0.6, color="#ff3b3b", alpha=0.08))
        ax.scatter([0], [0], c="#111111", s=90, marker="*", label="目标命题")
        if constraint_coords:
            ax.scatter([p[0] for p in constraint_coords], [p[1] for p in constraint_coords], c="#cc0000", s=28, alpha=0.6, label="约束条件")
        by_agent_path: Dict[str, List[Tuple[int, float, float]]] = {}
        for (agent, rnd, _), (x, y) in zip(point_meta, point_coords):
            by_agent_path.setdefault(agent, []).append((rnd, x, y))
        cmap = __import__("matplotlib").cm.get_cmap("tab10")
        for idx, (agent, pts) in enumerate(by_agent_path.items()):
            pts_sorted = sorted(pts, key=lambda x: x[0])
            xs = [p[1] for p in pts_sorted]
            ys = [p[2] for p in pts_sorted]
            ax.plot(xs, ys, color=cmap(idx % 10), linewidth=1.6, label=agent)
            ax.scatter(xs, ys, color=cmap(idx % 10), s=20)
        ax.set_title("观点势场：与约束红线距离")
        ax.set_xlabel("主成分1")
        ax.set_ylabel("主成分2")
        ax.axhline(0, color="#666", linewidth=0.6)
        ax.axvline(0, color="#666", linewidth=0.6)
        ax.legend(fontsize=8, loc="upper right")
        out_path = os.path.join(PLOTS_DIR, f"{base}_potential_field.jpg")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, format="jpg")
        plt.close(fig)
        self._log(f"[stance_eval] saved potential field: {out_path}")

    def plot_stance_history(
        self,
        history: List[Dict[str, Any]],
        round_logs: List[Dict[str, Any]],
        agent_names: List[str],
        language_mode: str,
        model_info: Dict[str, Any],
        rounds: int,
        event_id: str,
        value_weights_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        if str(os.environ.get("DISABLE_SIM_PLOTS", "0")).strip() == "1":
            self._log("[stance_eval] DISABLE_SIM_PLOTS=1, skip plotting/exports")
            return
        compact_output = False
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            self._log("[stance_eval] matplotlib not available, skip plotting")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        resolved = self.resolve_model_name(model_info or {}, fallback_llm=self.llm)
        base = f"{ts}_{rounds}_{language_mode}_{self._safe_name(resolved)}"
        rounds_sorted = sorted({r.get("round", 0) for r in round_logs}) if round_logs else list(range(1, rounds + 1))
        round_logs_sorted = sorted(round_logs, key=lambda r: r.get("round", 0)) if round_logs else []
        agent_series = self._series_from_rounds(round_logs_sorted, agent_names, infer_fn=lambda rlog, a: self._infer_stance_from_round(rlog, a, language_mode))
        injected_rounds = [
            int(r.get("round") or 0)
            for r in round_logs_sorted
            if bool(r.get("event_constraints_changed"))
            or (int(r.get("round") or 0) == 1 and bool(r.get("event_constraints")))
        ]

        fig1, ax1 = plt.subplots(figsize=(9, 5))
        pressures = self._pressure_by_round(round_logs_sorted, len(agent_names))
        if pressures:
            pmax = max(pressures) or 1.0
            for idx, r in enumerate(rounds_sorted[: len(pressures)]):
                alpha = min(0.25, max(0.02, pressures[idx] / pmax * 0.25))
                ax1.axvspan(r - 0.5, r + 0.5, color="#ff6b6b", alpha=alpha)
        for a in agent_names:
            series = agent_series.get(a, [])
            if not series:
                continue
            rs = rounds_sorted[: len(series)]
            ax1.plot(rs, series, label=a, linewidth=1.5)
            ax1.scatter(rs, series, s=12)
        for rr in sorted(set(injected_rounds)):
            ax1.axvline(rr, linestyle="--", linewidth=1.0, color="#8b5cf6", alpha=0.55)
        ax1.set_xlabel("轮次")
        ax1.set_ylabel("态度分值")
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_yticks(self.STANCE_TICKS)
        ax1.set_yticklabels(self.STANCE_LABELS, fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        ax1.set_title(f"态度变化趋势（{event_id}）")
        ax1.legend(fontsize=8)
        line_path = os.path.join(PLOTS_DIR, f"{base}.jpg")
        fig1.tight_layout()
        fig1.savefig(line_path, dpi=150, format="jpg")
        plt.close(fig1)

        heat_path = ""
        box_path = ""
        if not compact_output:
            fig2, ax2 = plt.subplots(figsize=(9, 4.8))
            if agent_names and round_logs:
                r_index = {rv: i for i, rv in enumerate(rounds_sorted)}
                a_index = {a: i for i, a in enumerate(agent_names)}
                matrix = [[0 for _ in rounds_sorted] for _ in agent_names]
                for rlog in round_logs:
                    ridx = r_index.get(rlog.get("round", 0))
                    for inf in rlog.get("influences", []) or []:
                        if not isinstance(inf, dict) or not inf.get("changed"):
                            continue
                        a = inf.get("agent")
                        if a in a_index and ridx is not None:
                            matrix[a_index[a]][ridx] = 1
                ax2.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
                ax2.set_yticks(range(len(agent_names)))
                ax2.set_yticklabels(agent_names)
                ax2.set_xticks(range(len(rounds_sorted)))
                ax2.set_xticklabels([f"R{r}" for r in rounds_sorted])
                ax2.set_title("影响变化热力图")
            else:
                ax2.text(0.1, 0.5, "No influence data", transform=ax2.transAxes)
            heat_path = os.path.join(PLOTS_DIR, f"{base}_heatmap.jpg")
            fig2.tight_layout()
            fig2.savefig(heat_path, dpi=150, format="jpg")
            plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            series_list = []
            labels = []
            for a in agent_names:
                vals = list(agent_series.get(a, []) or [0.0])
                series_list.append(vals)
                labels.append(a)
            bp = ax3.boxplot(series_list, labels=labels, patch_artist=True, medianprops={"color": "black", "linewidth": 1.8})
            cmap = plt.get_cmap("Set2")
            for i, patch in enumerate(bp.get("boxes", [])):
                patch.set(facecolor=cmap(i % 8), alpha=0.75, edgecolor="#333")
            for i, vals in enumerate(series_list):
                x = [i + 1 + ((j % 3) - 1) * 0.03 for j in range(len(vals))]
                ax3.scatter(x, vals, s=9, color="#222", alpha=0.35, zorder=3)
            ax3.set_ylim(-1.05, 1.05)
            ax3.set_ylabel("态度分值")
            ax3.set_yticks(self.STANCE_TICKS)
            ax3.set_yticklabels(self.STANCE_LABELS, fontsize=8)
            ax3.set_title("代理态度分布（箱线图）")
            ax3.grid(axis="y", linestyle="--", alpha=0.3)
            ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            box_path = os.path.join(PLOTS_DIR, f"{base}_boxplot.jpg")
            fig3.tight_layout()
            fig3.savefig(box_path, dpi=150, format="jpg")
            plt.close(fig3)

            self._plot_radar(round_logs_sorted, agent_names, base)
            self._plot_potential_field(round_logs_sorted, agent_names, base)
            self._save_io_exports(base)
            self._save_induction_feature_stats(round_logs_sorted, agent_names, agent_series, base)
            self._maybe_save_agent_profile_table(agent_names, base)
            self._plot_agent_stance_matrix(round_logs_sorted, agent_names, agent_series, base)

        runtime_store.set_state({
            "last_chart_event_id": event_id,
            "last_chart_path": line_path,
            "last_heatmap_path": heat_path,
            "last_boxplot_path": box_path,
            "last_plot_base": base,
        })
        self._log(f"[stance_eval] saved plot: {line_path}")
        if heat_path:
            self._log(f"[stance_eval] saved heatmap: {heat_path}")
        if box_path:
            self._log(f"[stance_eval] saved boxplot: {box_path}")
