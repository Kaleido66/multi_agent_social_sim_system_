#!/usr/bin/env python3
"""
L2: Statistical Analysis of 2×2 Ablation with Repetitions
===========================================================

Reads the merged longtable CSV produced by ``l2_runner.py`` (or the existing
``run_ablation_4way.py --repeats 5``) and performs:

  1. Per-agent × per-condition descriptive statistics
     (mean stance, SD, 95% bootstrap CI)
  2. Paired comparisons between conditions
     (paired t-test / Wilcoxon, Cohen's d)
  3. Mixed-effects model (optional, requires statsmodels)
  4. Publication-ready tables (CSV + printed markdown)

Input
-----
  A CSV with at minimum these columns:
    agent, nationality, faction, module_combo, induction_on, feature_on,
    round, metric_value, stance_label, repeat_index

Outputs
-------
  data/experiments/l2_agent_summary.csv       – per-agent descriptive stats
  data/experiments/l2_condition_contrasts.csv  – paired condition comparisons
  data/experiments/l2_effect_sizes.csv         – Cohen's d table
  (printed)  Markdown tables for the paper

Usage
-----
  # After running l2_runner.py:
  python experiments/l2_statistics.py --csv data/experiments/l2_merged_longtable.csv

  # Or point at any ablation longtable CSV with repeat_index column:
  python experiments/l2_statistics.py --csv data/logs/ablation_2x2_longtable_xxx.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "data", "experiments")

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_longtable(csv_path: str) -> List[Dict[str, Any]]:
    """Load the ablation longtable CSV into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        return rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # numeric conversions
            for key in ("round", "repeat_index", "induction_on", "feature_on"):
                try:
                    row[key] = int(row.get(key, 0) or 0)
                except (ValueError, TypeError):
                    row[key] = 0
            try:
                row["metric_value"] = float(row.get("metric_value", 0) or 0)
            except (ValueError, TypeError):
                row["metric_value"] = 0.0
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# grouping helpers
# ---------------------------------------------------------------------------

MODULE_COMBO_ORDER = [
    "ind_off__feat_off",
    "ind_off__feat_on",
    "ind_on__feat_off",
    "ind_on__feat_on",
]

MODULE_COMBO_LABELS = {
    "ind_off__feat_off":  "B (双关)",
    "ind_off__feat_on":   "C (仅约束)",
    "ind_on__feat_off":   "D (仅诱导)",
    "ind_on__feat_on":    "A (双开/基线)",
}


def group_by_agent_combo(
    rows: List[Dict[str, Any]],
    final_round_only: bool = False,
) -> Dict[Tuple[str, str], List[float]]:
    """
    Group metric_value by (agent, module_combo).

    If final_round_only=True, only use rows where is_final_round==1.
    Otherwise, use all rounds (for trajectory-level statistics).
    """
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        if final_round_only and int(r.get("is_final_round", 0)) != 1:
            continue
        agent = str(r.get("agent", ""))
        combo = str(r.get("module_combo", ""))
        groups[(agent, combo)].append(float(r["metric_value"]))
    return dict(groups)


def per_repeat_means(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, int], float]:
    """
    Compute per-(agent, combo, repeat_index) mean stance.
    This is the unit of analysis for paired tests across conditions.
    """
    groups: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for r in rows:
        agent = str(r.get("agent", ""))
        combo = str(r.get("module_combo", ""))
        rep = int(r.get("repeat_index", 1))
        groups[(agent, combo, rep)].append(float(r["metric_value"]))
    means: Dict[Tuple[str, str, int], float] = {}
    for key, vals in groups.items():
        means[key] = sum(vals) / len(vals)
    return means


# ---------------------------------------------------------------------------
# descriptive statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(data: List[float], n_bootstrap: int = 2000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap percentile confidence interval for the mean."""
    import random as _random
    _random.seed(seed)
    n = len(data)
    if n < 3:
        m = sum(data) / n if n > 0 else 0.0
        return m, m
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [_random.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    return means[max(0, lo_idx)], means[min(n_bootstrap - 1, hi_idx)]


def compute_agent_summary(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-agent × per-condition descriptive stats."""
    import statistics
    groups = group_by_agent_combo(rows, final_round_only=False)
    summary: List[Dict[str, Any]] = []
    for (agent, combo), vals in sorted(groups.items()):
        if len(vals) < 2:
            continue
        mean_ = statistics.mean(vals)
        sd_ = statistics.stdev(vals)
        ci_lo, ci_hi = bootstrap_ci(vals)
        # agent metadata from first matching row
        meta = {}
        for r in rows:
            if r.get("agent") == agent:
                meta = r
                break
        summary.append({
            "agent": agent,
            "nationality": str(meta.get("nationality", "")),
            "faction": str(meta.get("political_identity", "")),
            "module_combo": combo,
            "combo_label": MODULE_COMBO_LABELS.get(combo, combo),
            "n": len(vals),
            "mean_stance": round(mean_, 4),
            "sd": round(sd_, 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
        })
    return summary


# ---------------------------------------------------------------------------
# paired comparisons
# ---------------------------------------------------------------------------


def cohens_d(x: List[float], y: List[float]) -> float:
    """Cohen's d for paired samples."""
    import math
    n = len(x)
    if n < 2:
        return 0.0
    diffs = [x[i] - y[i] for i in range(n)]
    mean_d = sum(diffs) / n
    if mean_d == 0:
        return 0.0
    sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    if sd_d == 0:
        return float("inf") if mean_d > 0 else float("-inf")
    return mean_d / sd_d


def paired_ttest(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Paired t-test.  Returns (t_statistic, p_value)."""
    import math
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    diffs = [x[i] - y[i] for i in range(n)]
    mean_d = sum(diffs) / n
    sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    if sd_d == 0:
        return float("inf") if mean_d != 0 else 0.0, 0.0 if mean_d != 0 else 1.0
    se = sd_d / math.sqrt(n)
    t = mean_d / se
    # use survival function approximation via stdlib
    # (for proper t-distribution CDF, scipy would be needed; we use a
    #  normal-approximation p-value annotated with * markers)
    import math as _m
    # two-tailed p via normal approx
    p = 2 * (1.0 - _normal_cdf(abs(t)))
    return t, min(p, 1.0)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    import math
    if x < 0:
        return 1.0 - _normal_cdf(-x)
    # constants
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p_val = 0.3275911
    t = 1.0 / (1.0 + p_val * x)
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)
    return y


def compute_condition_contrasts(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each agent, compare every pair of conditions using per-repeat means
    as the unit of analysis.  Reports Δ, t-stat, p-value, Cohen's d.
    """
    rep_means = per_repeat_means(rows)
    # pivot to agent -> combo -> [mean across repeats]
    pivot: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (agent, combo, rep), val in sorted(rep_means.items()):
        pivot[agent][combo].append(val)

    combos = MODULE_COMBO_ORDER
    contrast_pairs = [
        ("ind_on__feat_on", "ind_off__feat_off", "A vs B (双开 vs 双关)"),
        ("ind_on__feat_off", "ind_off__feat_off", "D vs B (仅诱导 vs 双关)"),
        ("ind_off__feat_on", "ind_off__feat_off", "C vs B (仅约束 vs 双关)"),
        ("ind_on__feat_on", "ind_off__feat_on", "A vs C (双开 vs 仅约束)"),
        ("ind_on__feat_on", "ind_on__feat_off", "A vs D (双开 vs 仅诱导)"),
        ("ind_on__feat_off", "ind_off__feat_on", "D vs C (仅诱导 vs 仅约束)"),
    ]

    results: List[Dict[str, Any]] = []
    for agent in sorted(pivot.keys()):
        agent_data = pivot[agent]
        for c1, c2, label in contrast_pairs:
            x = agent_data.get(c1, [])
            y = agent_data.get(c2, [])
            if len(x) < 2 or len(y) < 2:
                continue
            # align by repeat index if lengths differ
            n = min(len(x), len(y))
            x_aligned = x[:n]
            y_aligned = y[:n]
            mean_diff = sum(x_aligned[i] - y_aligned[i] for i in range(n)) / n
            t_stat, p_val = paired_ttest(x_aligned, y_aligned)
            d = cohens_d(x_aligned, y_aligned)
            # significance markers
            sig = ""
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            # metadata
            meta = {}
            for r_ in rows:
                if r_.get("agent") == agent:
                    meta = r_
                    break
            results.append({
                "agent": agent,
                "nationality": str(meta.get("nationality", "")),
                "faction": str(meta.get("political_identity", "")),
                "contrast": label,
                "combo_1": c1,
                "combo_2": c2,
                "mean_1": round(sum(x_aligned) / n, 4),
                "mean_2": round(sum(y_aligned) / n, 4),
                "delta": round(mean_diff, 4),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_val, 6),
                "p_significance": sig,
                "cohens_d": round(d, 4),
                "n_pairs": n,
            })
    return results


# ---------------------------------------------------------------------------
# mixed effects model (optional)
# ---------------------------------------------------------------------------


def try_mixed_effects(rows: List[Dict[str, Any]]) -> Optional[str]:
    """
    Attempt a mixed-effects model:
      metric_value ~ induction_on * feature_on + (1 | agent) + (1 | round)

    Requires ``statsmodels``.  Returns a formatted summary string or None.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        return None

    # build DataFrame
    data_rows = []
    for r in rows:
        data_rows.append({
            "metric_value": float(r["metric_value"]),
            "induction_on": int(r.get("induction_on", 0)),
            "feature_on": int(r.get("feature_on", 0)),
            "agent": str(r.get("agent", "")),
            "round": int(r.get("round", 0)),
            "interaction": int(r.get("induction_on", 0)) * int(r.get("feature_on", 0)),
        })
    if len(data_rows) < 20:
        return "Too few rows for mixed-effects model (need ≥20)"

    import pandas as pd
    df = pd.DataFrame(data_rows)

    try:
        model = MixedLM(
            endog=df["metric_value"],
            exog=df[["induction_on", "feature_on", "interaction"]],
            groups=df["agent"],
            # exog_re=df[["round"]],
        )
        result = model.fit(reml=True, maxiter=100)
        return result.summary().as_text()
    except Exception as exc:
        return f"Mixed-effects model failed: {exc}"


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def save_csv(rows: List[Dict[str, Any]], path: str, headers: List[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_markdown_table(rows: List[Dict[str, Any]],
                         columns: List[str],
                         col_labels: List[str],
                         title: str = "") -> None:
    """Print a GitHub-flavoured markdown table."""
    print(f"\n### {title}\n")
    header = "| " + " | ".join(col_labels) + " |"
    sep = "|" + "|".join([" --- " for _ in col_labels]) + "|"
    print(header)
    print(sep)
    for r in rows:
        vals = [str(r.get(c, "")) for c in columns]
        print("| " + " | ".join(vals) + " |")
    print()


def print_agent_summary_markdown(summary: List[Dict[str, Any]]) -> None:
    """Print the per-agent summary as a paper-ready table."""
    columns = ["agent", "nationality", "combo_label", "mean_stance", "ci_95_lo", "ci_95_hi", "n"]
    col_labels = ["Agent", "Nationality", "Condition", "Mean Stance",
                  "CI Low", "CI High", "N"]
    print_markdown_table(summary, columns, col_labels,
                         "Per-Agent Descriptive Statistics")


def print_contrasts_markdown(contrasts: List[Dict[str, Any]]) -> None:
    """Print the key condition contrasts."""
    # filter to the most important contrasts
    key_contrasts = ["A vs B (双开 vs 双关)", "D vs B (仅诱导 vs 双关)",
                     "C vs B (仅约束 vs 双关)"]
    filtered = [c for c in contrasts if c["contrast"] in key_contrasts]
    columns = ["agent", "nationality", "contrast", "delta", "cohens_d",
               "p_significance", "n_pairs"]
    col_labels = ["Agent", "Nat.", "Contrast", "Δ Stance",
                  "Cohen's d", "Sig.", "N"]
    print_markdown_table(filtered, columns, col_labels,
                         "Key Condition Contrasts per Agent")


def print_effect_size_table(contrasts: List[Dict[str, Any]]) -> None:
    """Print a pivot table of Cohen's d (agent × contrast)."""
    agents = sorted(set(c["agent"] for c in contrasts))
    key_contrasts = ["A vs B (双开 vs 双关)", "D vs B (仅诱导 vs 双关)",
                     "C vs B (仅约束 vs 双关)"]
    # pivot
    d_map: Dict[Tuple[str, str], float] = {}
    for c in contrasts:
        if c["contrast"] in key_contrasts:
            d_map[(c["agent"], c["contrast"])] = c["cohens_d"]

    print("\n### Cohen's d Effect Sizes (agent × contrast)\n")
    header = "| Agent | " + " | ".join(key_contrasts) + " |"
    sep = "| --- " * (len(key_contrasts) + 1) + "|"
    print(header)
    print(sep)
    for agent in agents:
        vals = []
        for kc in key_contrasts:
            d = d_map.get((agent, kc), 0)
            vals.append(f"{d:+.2f}")
        print(f"| {agent} | " + " | ".join(vals) + " |")
    print()


def print_hypothesis_tests(contrasts: List[Dict[str, Any]],
                           summary: List[Dict[str, Any]]) -> None:
    """Print a structured summary addressing each pre-registered hypothesis."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST SUMMARY")
    print("=" * 70)

    # H1: Induction effect
    print("\n**H1 (Induction Effect):** Induction significantly shifts target agents' stance.")
    h1_contrasts = [c for c in contrasts if c["contrast"] == "D vs B (仅诱导 vs 双关)"]
    for c in h1_contrasts:
        print(f"  {c['agent']:20s}  Δ={c['delta']:+.3f}  d={c['cohens_d']:+.2f}  "
              f"p={c['p_value']:.4f} {c['p_significance']}")

    # H2: Constraint effect
    print("\n**H2 (Constraint Effect):** Progressive constraints suppress extreme stances.")
    h2_contrasts = [c for c in contrasts if c["contrast"] == "C vs B (仅约束 vs 双关)"]
    for c in h2_contrasts:
        print(f"  {c['agent']:20s}  Δ={c['delta']:+.3f}  d={c['cohens_d']:+.2f}  "
              f"p={c['p_value']:.4f} {c['p_significance']}")

    # H3: Interaction
    print("\n**H3 (Interaction Effect):** Induction and constraints show antagonistic interaction.")
    h3_contrasts = [c for c in contrasts if c["contrast"] == "A vs B (双开 vs 双关)"]
    for c in h3_contrasts:
        print(f"  {c['agent']:20s}  Δ={c['delta']:+.3f}  d={c['cohens_d']:+.2f}  "
              f"p={c['p_value']:.4f} {c['p_significance']}")

    # H4: Defense rebound  (partial support expected)
    print("\n**H4 (Defense Rebound):** Some agents show defensive resistance to induction.")
    # Look for agents with large negative Δ in C vs D comparison
    defense_contrasts = [c for c in contrasts if c["contrast"] == "D vs C (仅诱导 vs 仅约束)"]
    for c in defense_contrasts:
        print(f"  {c['agent']:20s}  Δ={c['delta']:+.3f}  d={c['cohens_d']:+.2f}  "
              f"p={c['p_value']:.4f} {c['p_significance']}")

    # H5: Polarization dynamics
    print("\n**H5 (Polarization Dynamics):** Induction tends to ↑polarization, constraints ↓polarization.")
    print("  (Requires group-level polarization metrics from round_metrics – see digest data)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L2: Statistical analysis of ablation experiment repetitions"
    )
    p.add_argument("--csv", type=str, required=True,
                   help="Path to merged longtable CSV (with repeat_index column)")
    p.add_argument("--out-dir", type=str, default=OUT_DIR,
                   help="Output directory for result CSVs")
    p.add_argument("--mixed-effects", action="store_true",
                   help="Attempt mixed-effects model (requires statsmodels)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_longtable(args.csv)
    if not rows:
        return 1

    repeats_found = len(set(r.get("repeat_index", 0) for r in rows))
    agents_found = sorted(set(r.get("agent", "") for r in rows))
    combos_found = sorted(set(r.get("module_combo", "") for r in rows))
    print(f"  repeats={repeats_found}  agents={len(agents_found)}  "
          f"combos={len(combos_found)}")
    print(f"  agents: {agents_found}")
    print(f"  combos: {combos_found}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Agent summary
    summary = compute_agent_summary(rows)
    summary_path = save_csv(
        summary,
        os.path.join(args.out_dir, f"l2_agent_summary_{ts}.csv"),
        ["agent", "nationality", "faction", "module_combo", "combo_label",
         "n", "mean_stance", "sd", "ci_95_lo", "ci_95_hi"],
    )
    print(f"\nAgent summary: {summary_path}  ({len(summary)} rows)")

    # 2) Condition contrasts
    contrasts = compute_condition_contrasts(rows)
    contrasts_path = save_csv(
        contrasts,
        os.path.join(args.out_dir, f"l2_condition_contrasts_{ts}.csv"),
        ["agent", "nationality", "faction", "contrast", "combo_1", "combo_2",
         "mean_1", "mean_2", "delta", "t_stat", "p_value", "p_significance",
         "cohens_d", "n_pairs"],
    )
    print(f"Condition contrasts: {contrasts_path}  ({len(contrasts)} rows)")

    # 3) Mixed effects (optional)
    if args.mixed_effects:
        me_result = try_mixed_effects(rows)
        if me_result:
            me_path = os.path.join(args.out_dir, f"l2_mixed_effects_{ts}.txt")
            with open(me_path, "w", encoding="utf-8") as f:
                f.write(me_result)
            print(f"Mixed effects: {me_path}")
        else:
            print("Mixed effects: statsmodels not available (install with: pip install statsmodels)")

    # 4) Print paper-ready tables
    print_agent_summary_markdown(summary)
    print_contrasts_markdown(contrasts)
    print_effect_size_table(contrasts)
    print_hypothesis_tests(contrasts, summary)

    # 5) Save summary JSON
    summary_json = {
        "timestamp": ts,
        "csv_source": args.csv,
        "n_rows": len(rows),
        "n_repeats": repeats_found,
        "n_agents": len(agents_found),
        "agents": agents_found,
        "n_combos": len(combos_found),
        "combos": combos_found,
        "significant_contrasts": [
            {"agent": c["agent"], "contrast": c["contrast"], "delta": c["delta"],
             "d": c["cohens_d"], "p": c["p_value"]}
            for c in contrasts if c["p_value"] < 0.05
        ],
    }
    json_path = os.path.join(args.out_dir, f"l2_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"\nSummary JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
