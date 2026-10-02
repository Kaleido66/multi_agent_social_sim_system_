#!/usr/bin/env python3
"""Statistics and plotting for the L1 equivalence-path experiment.

Three conditions are compared, all on stance_score = support - oppose:

  arm0_no_persona   -- raw backbone model, no persona content (runner JSONL).
                       Persona-independent, so it is a single baseline per model
                       that gets broadcast against every persona in comparisons().
  arm1_persona      -- persona description only, no framework. This is the
                       "网页端直接对话对照" (web direct-dialogue control):
                       one independent stateless call per run.
  arm2_full_framework -- the full production simulation with induction ON and
                       features ON ("诱导+特征+"), read from each model's
                       both_on digest file. Round 0 is excluded.

Caveat on arm2's sample size: its rounds come from a single stateful simulation
run per model, so the 20 rounds are autocorrelated rather than 20 independent
draws. They are still treated as 20 samples here (that is the only within-run
variation available), which overstates arm2's effective sample size and makes
arm1-vs-arm2 p-values optimistic. Read the rank-biserial effect sizes as the
primary signal; treat borderline p-values with caution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.role_library import BASE_ROLES  # noqa: E402


DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "experiments", "equivalence_path_l1")
LOG_DIR = os.path.join(ROOT, "data", "logs")

NAME_ALIASES = {
    "刘宇": {"刘宇", "Liu Yu"},
    "顾平": {"顾平", "Gu Ping"},
    "张强": {"张强", "Zhang Qiang"},
    "李建国": {"李建国", "Li Jianguo"},
    "陈思远": {"陈思远", "Chen Siyuan"},
}
for _role in BASE_ROLES:
    NAME_ALIASES.setdefault(str(_role.get("name", "")), {str(_role.get("name", ""))})

# English display names for personas whose original name is Chinese, used for plot labels.
NAME_EN = {
    "刘宇": "Liu Yu",
    "顾平": "Gu Ping",
    "张强": "Zhang Qiang",
    "李建国": "Li Jianguo",
    "陈思远": "Chen Siyuan",
}

# arm2 = the full production framework (induction ON + feature ON), read directly
# from each model's "both_on" digest file. This is the "网页端直接对话对照"
# reference: does persona-only direct dialogue (arm1) reproduce the full
# simulation (arm2)? The model_alias -> digest-filename mapping is copied from
# export_attitude_excel.py's MODEL_CONFIGS (the "诱导+特征+" condition).
ARM2_DIGESTS = {
    "qwen3.6-27b": "run_20260630_102348_zh_ecnu-plus_digest.json",
    "deepseek-v4-flash": "run_20260811_105848_zh_ecnu-max_digest.json",
    "llama-3.3-70b": "run_20260805_112224_zh_meta-llama_llama-3.3-70b-instruct_digest.json",
    "gpt-5.6-terra": "run_20260805_124918_zh_openai_gpt-5.6-terra_digest.json",
    "gemini-3-flash": "run_20260805_195427_zh_google_gemini-3-flash-preview_digest.json",
    "claude-sonnet-4.6": "run_20260810_141635_zh_anthropic_claude-sonnet-4.6_digest.json",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def bootstrap_ci(values: List[float], n_bootstrap: int = 5000, seed: int = 20260914) -> Tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (n_bootstrap - 1))]
    hi = means[int(0.975 * (n_bootstrap - 1))]
    return (lo, hi)


def per_run_metrics(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    meta: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        score = safe_float(row.get("stance_score"))
        if score is None:
            continue
        key = (
            str(row.get("model_alias") or row.get("model_name") or ""),
            str(row.get("condition") or ""),
            str(row.get("persona_id") or ""),
            str(row.get("run_id") or ""),
        )
        grouped[key].append(score)
        meta[key] = row
    out = []
    for key, scores in sorted(grouped.items()):
        model, condition, persona_id, run_id = key
        first = meta[key]
        out.append({
            "model_alias": model,
            "condition": condition,
            "persona_id": persona_id,
            "persona_name": first.get("persona_name", ""),
            "run_id": run_id,
            "n_rounds": len(scores),
            "mean_stance": sum(scores) / len(scores),
            "sigma": stdev(scores),
        })
    return out


def summarize_runs(run_rows: Iterable[Dict[str, Any]], n_bootstrap: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["model_alias"], row["condition"], row["persona_id"])].append(row)
    out = []
    for (model, condition, persona_id), rows in sorted(grouped.items()):
        means = [float(r["mean_stance"]) for r in rows]
        mlo, mhi = bootstrap_ci(means, n_bootstrap=n_bootstrap)
        out.append({
            "model_alias": model,
            "condition": condition,
            "persona_id": persona_id,
            "persona_name": rows[0].get("persona_name", ""),
            "n_runs": len(rows),
            "mean_stance": sum(means) / len(means),
            "mean_stance_ci_low": mlo,
            "mean_stance_ci_high": mhi,
            # Dispersion ACROSS runs/rounds. This is the meaningful spread here:
            # a within-unit sigma is degenerate by construction, because every
            # unit is itself a single sample (one arm0/arm1 call, or one arm2
            # round) and so always has n_rounds == 1 and sigma == 0.
            "sd_across_runs": stdev(means),
        })
    return out


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_whitney_u(x: List[float], y: List[float]) -> Dict[str, float]:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return {"u": math.nan, "p_value": math.nan, "rank_biserial": math.nan}
    values = [(v, 0) for v in x] + [(v, 1) for v in y]
    values.sort(key=lambda item: item[0])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_x = sum(rank for rank, item in zip(ranks, values) if item[1] == 0)
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = 0.0 if sd_u == 0 else (u1 - mean_u) / sd_u
    p = 2.0 * min(normal_cdf(z), 1.0 - normal_cdf(z))
    rbc = (2.0 * u1 / (n1 * n2)) - 1.0
    return {"u": u1, "p_value": p, "rank_biserial": rbc}


def arm2_rows_from_digests(log_dir: str) -> List[Dict[str, Any]]:
    """Extract arm2 (full framework) stance samples from each model's "both_on"
    digest. Chinese persona names are garbled inside the digest's stance_eval
    keys, so agents are matched positionally against BASE_ROLES (their order is
    identical). Each simulation round is treated as one stance sample."""
    out: List[Dict[str, Any]] = []
    for model_alias, fname in ARM2_DIGESTS.items():
        path = os.path.join(log_dir, fname)
        if not os.path.exists(path):
            print(f"[warn] arm2 digest not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rounds = data["events"][0]["rounds"]
        for r in rounds:
            rn = int(r.get("round", -1))
            if rn < 1:
                continue  # skip round 0 (initialization) and invalid rounds
            stance_eval = r.get("stance_eval") or {}
            if not isinstance(stance_eval, dict) or len(stance_eval) != len(BASE_ROLES):
                continue
            keys = list(stance_eval.keys())  # garbled order matches BASE_ROLES order
            for i, role in enumerate(BASE_ROLES):
                ev = stance_eval.get(keys[i])
                if not isinstance(ev, dict):
                    continue
                stance = float(ev.get("support", 0.0)) - float(ev.get("oppose", 0.0))
                out.append({
                    "model_alias": model_alias,
                    "condition": "arm2_full_framework",
                    "persona_id": str(role.get("id")),
                    "persona_name": role.get("name", ""),
                    "run_id": str(rn),
                    "n_rounds": 1,
                    "mean_stance": stance,
                    "sigma": 0.0,
                })
    return out


def comparisons(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["model_alias"], row["persona_id"], row["condition"])].append(float(row["mean_stance"]))
    # arm0_no_persona is persona-independent: a single global baseline per model,
    # so collect it once per model and broadcast it against every persona's arm1.
    arm0_by_model: Dict[str, List[float]] = defaultdict(list)
    for (model, persona_id, condition), vals in grouped.items():
        if condition == "arm0_no_persona":
            arm0_by_model[model].extend(vals)
    out = []
    keys = sorted({(m, p) for (m, p, _c) in grouped if p})
    for model, persona_id in keys:
        arm0 = arm0_by_model.get(model, [])
        arm1 = grouped.get((model, persona_id, "arm1_persona"), [])
        arm2 = grouped.get((model, persona_id, "arm2_full_framework"), [])
        if arm0 and arm1:
            stat = mann_whitney_u(arm0, arm1)
            out.append({"model_alias": model, "persona_id": persona_id, "comparison": "arm0_vs_arm1", **stat})
        if arm1 and arm2:
            stat = mann_whitney_u(arm1, arm2)
            out.append({"model_alias": model, "persona_id": persona_id, "comparison": "arm1_vs_arm2", **stat})
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: List[Dict[str, Any]], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    markers = {"arm1_persona": "s", "arm2_full_framework": "^"}
    colors = {"arm0_no_persona": "#4C78A8", "arm1_persona": "#F58518", "arm2_full_framework": "#54A24B"}
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_model[row["model_alias"]].append(row)
    models = sorted(by_model)
    personas = [str(r.get("id")) for r in BASE_ROLES]
    labels = [NAME_EN.get(str(r.get("name")), str(r.get("name"))) for r in BASE_ROLES]

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        rows = by_model[model]

        # arm0_no_persona is persona-independent: flat baseline line + CI band.
        arm0 = next((r for r in rows if r["condition"] == "arm0_no_persona"), None)
        if arm0 is not None:
            mean = float(arm0["mean_stance"])
            lo = float(arm0["mean_stance_ci_low"])
            hi = float(arm0["mean_stance_ci_high"])
            xmin, xmax = -0.5, len(personas) - 0.5
            ax.axhline(mean, color=colors["arm0_no_persona"], linewidth=1.2)
            ax.fill_between([xmin, xmax], lo, hi, color=colors["arm0_no_persona"], alpha=0.15)

        # arm1 / arm2 are per-persona: error bars with bootstrap CI.
        for offset, condition in zip([-0.12, 0.12], ["arm1_persona", "arm2_full_framework"]):
            xs, ys, yerr_low, yerr_high = [], [], [], []
            for idx, persona_id in enumerate(personas):
                item = next((r for r in rows if r["condition"] == condition and r["persona_id"] == persona_id), None)
                if not item:
                    continue
                mean = float(item["mean_stance"])
                xs.append(idx + offset)
                ys.append(mean)
                yerr_low.append(mean - float(item["mean_stance_ci_low"]))
                yerr_high.append(float(item["mean_stance_ci_high"]) - mean)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_low, yerr_high],
                    fmt=markers[condition],
                    color=colors[condition],
                    capsize=2,
                    markersize=4,
                    linestyle="none",
                )
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_title(model, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes[len(models):]:
        ax.axis("off")

    legend_handles = [
        Line2D([], [], color=colors["arm0_no_persona"], linewidth=1.2, label="arm0_no_persona"),
        Line2D([], [], color=colors["arm1_persona"], marker=markers["arm1_persona"], linestyle="none", label="arm1_persona"),
        Line2D([], [], color=colors["arm2_full_framework"], marker=markers["arm2_full_framework"], linestyle="none", label="arm2_full_framework"),
    ]
    fig.suptitle("Equivalence Path L1: mean stance by persona and condition", y=0.995, fontsize=11)
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=3, frameon=False, fontsize=8,
    )
    fig.supylabel("mean stance score")
    fig.tight_layout(rect=(0.005, 0, 1, 0.935))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize L1 equivalence-path results.")
    parser.add_argument("--jsonl", required=True, help="Runner JSONL output.")
    parser.add_argument("--log-dir", default=LOG_DIR, help="Directory holding arm2 (full-framework) digest files.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_rows = read_jsonl(args.jsonl)
    run_rows = per_run_metrics(raw_rows)
    run_rows.extend(arm2_rows_from_digests(args.log_dir))
    summary = summarize_runs(run_rows, n_bootstrap=args.bootstrap)
    comp = comparisons(run_rows)
    write_csv(os.path.join(args.out_dir, "l1_per_run_metrics.csv"), run_rows)
    write_csv(os.path.join(args.out_dir, "l1_summary.csv"), summary)
    write_csv(os.path.join(args.out_dir, "l1_arm_comparisons.csv"), comp)
    plot_summary(summary, os.path.join(args.out_dir, "l1_mean_stance_ci.png"))
    print(f"[done] wrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
