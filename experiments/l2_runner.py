#!/usr/bin/env python3
"""
L2: Repetition Runner for 2×2 Ablation (self-contained)
========================================================

Replicates the manual ablation workflow:
  1. rename data/inductions away     → induction OFF
  2. rename data/event_constraints away → feature constraints OFF
  3. run main.py
  4. rename everything back
  5. parse the digest → longtable rows

Repeats the full 2×2 matrix N times with different random seeds for
statistical power.

Key design decisions
--------------------
- Uses ``os.rename()`` (atomic on a single drive) instead of ``shutil.move``.
  This avoids the "move across drives" / file-locking issues of the old script.
- Imports digest-parsing helpers from ``scripts.run_ablation_4way`` (importing
  is NOT modifying – it just reuses existing, tested logic).
- Calls ``main.py`` as a subprocess so every run gets a clean interpreter
  state (fresh imports, no cached config, isolated memory store).
- Does NOT depend on ``run_ablation_4way.py`` as an orchestrator.

Usage
-----
  python experiments/l2_runner.py
  python experiments/l2_runner.py --repeats 5 --seed 123
  python experiments/l2_runner.py --repeats 3 --timeout 1800 --only-combo A

Outputs
-------
  data/experiments/l2_runs/                    per-repeat logs + CSVs
  data/experiments/l2_merged_longtable.csv      unified dataset
  data/experiments/l2_manifest.jsonl            run metadata
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))   # needed for ablation_naming import

# ---------------------------------------------------------------------------
# Import digest-parsing utilities from the existing script (import ≠ modify)
# ---------------------------------------------------------------------------
from scripts.run_ablation_4way import (         # noqa: E402
    extract_long_rows_from_digest,
    find_latest_digest,
    find_related_run_json,
    extract_agent_meta_from_run_json,
    parse_model_from_digest_path,
    write_longtable,
    append_jsonl as _append_jsonl_orig,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
INDUCTIONS_DIR = os.path.join(DATA_DIR, "inductions")
EVENT_CONSTRAINTS_DIR = os.path.join(DATA_DIR, "event_constraints")
MAIN_SCRIPT = os.path.join(ROOT, "main.py")
CLEAR_MEM_SCRIPT = os.path.join(ROOT, "scripts", "clear_agent_memories.py")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "experiments", "l2_runs")

# 2×2 combos in execution order: B → C → D → A
COMBO_SPECS = [
    {"induction": False, "feature": False, "tag": "ind_off__feat_off", "label": "B (双关)"},
    {"induction": False, "feature": True,  "tag": "ind_off__feat_on",  "label": "C (仅约束)"},
    {"induction": True,  "feature": False, "tag": "ind_on__feat_off",  "label": "D (仅诱导)"},
    {"induction": True,  "feature": True,  "tag": "ind_on__feat_on",   "label": "A (双开/基线)"},
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["SIM_RANDOM_SEED"] = str(seed)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# directory toggling  (the core of "manual workflow" automation)
# ---------------------------------------------------------------------------

@dataclass
class DirToggle:
    """Track one toggled directory so we can restore it later."""
    original: str          # e.g. D:\...\data\inductions
    backup: str            # e.g. D:\...\data\inductions._off
    was_disabled: bool = False


def _rmtree_retry(path: str, attempts: int = 5, delay: float = 0.3) -> bool:
    """Remove a directory tree with retries (Windows file-locking workaround)."""
    if not os.path.exists(path):
        return True
    for i in range(attempts):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return True
        except OSError:
            if i < attempts - 1:
                time.sleep(delay)
    return False


def toggle_off(dir_path: str) -> DirToggle:
    """
    Rename *dir_path* → *dir_path._off* so the simulation sees nothing.
    Returns a DirToggle that can be passed to toggle_on() later.

    If the directory doesn't exist, does nothing (was_disabled=False).
    If a ._off backup already exists (crashed previous run), restores it first.
    """
    original = os.path.abspath(dir_path)
    backup = original + "._off"
    dt = DirToggle(original=original, backup=backup, was_disabled=False)

    if os.path.exists(backup):
        # Stale backup from a previous crash – restore first, then re-disable.
        print(f"  [toggle] cleaning up stale backup: {backup}")
        _rmtree_retry(original)
        try:
            os.rename(backup, original)
        except OSError:
            pass  # will try again below

    if not os.path.exists(original):
        return dt  # nothing to disable

    os.rename(original, backup)
    dt.was_disabled = True
    return dt


def toggle_on(dt: DirToggle) -> None:
    """Restore a directory previously toggled off by toggle_off()."""
    if not dt.was_disabled:
        return
    # The simulation may have recreated the *original* as an empty directory
    # (InductionLibrary.__init__ does os.makedirs).  Remove it.
    _rmtree_retry(dt.original)
    if os.path.exists(dt.original):
        print(f"  [toggle] WARNING: could not remove recreated {dt.original}")
        # fallback: move recreated dir out of the way
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.rename(dt.original, dt.original + f".recreated_{ts}")

    try:
        os.rename(dt.backup, dt.original)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to restore {dt.backup} → {dt.original}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------

def run_python(script_path: str, timeout_s: int,
               extra_env: Optional[Dict[str, str]] = None,
               description: str = "") -> Tuple[int, str]:
    """Run a Python script as subprocess.  Returns (returncode, stdout+stderr)."""
    cmd = [sys.executable, script_path]
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    label = f" ({description})" if description else ""
    print(f"  [subprocess] {os.path.basename(script_path)}{label}")
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env,
            timeout=timeout_s or None,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode, output
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return 124, output


# ---------------------------------------------------------------------------
# single-run orchestration
# ---------------------------------------------------------------------------

def run_one_condition(
    experiment_id: str,
    induction_on: bool,
    feature_on: bool,
    combo_tag: str,
    repeat_index: int,
    timeout_s: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute ONE condition (e.g. "induction=ON, feature=OFF") end-to-end.

    Returns (run_record, longtable_rows).
    """
    tag = combo_tag
    start = now_iso()
    toggles: List[DirToggle] = []

    # 1) clear agent memories
    t0 = time.time()
    rc_mem, out_mem = run_python(CLEAR_MEM_SCRIPT, timeout_s=120,
                                 description="clear memories")

    # 2) toggle directories for this condition
    try:
        if not induction_on:
            dt = toggle_off(INDUCTIONS_DIR)
            if dt.was_disabled:
                toggles.append(dt)
        if not feature_on:
            dt = toggle_off(EVENT_CONSTRAINTS_DIR)
            if dt.was_disabled:
                toggles.append(dt)

        # 3) run main.py
        rc_main, out_main = run_python(MAIN_SCRIPT, timeout_s=timeout_s,
                                       description=f"repeat={repeat_index} {tag}")
    finally:
        # ALWAYS restore, even if main.py crashes
        for dt in reversed(toggles):
            try:
                toggle_on(dt)
            except Exception as exc:
                print(f"  [toggle] ERROR restoring {dt.original}: {exc}")

    # 4) find and parse digest
    digest_path = find_latest_digest(t0)
    model_name = parse_model_from_digest_path(digest_path or "")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    related_run_json = find_related_run_json(digest_path or "")
    agent_meta = extract_agent_meta_from_run_json(related_run_json or "")

    rows = extract_long_rows_from_digest(
        digest_path=digest_path or "",
        experiment_id=experiment_id,
        run_tag=tag,
        induction_on=induction_on,
        feature_on=feature_on,
        agent_meta=agent_meta,
    ) if digest_path else []

    for row in rows:
        row["model_name"] = model_name
        row["run_timestamp"] = run_timestamp
        row["repeat_index"] = repeat_index

    status = "ok" if (rc_main == 0 and bool(digest_path)) else "failed"
    rec = {
        "experiment_id": experiment_id,
        "repeat_index": repeat_index,
        "tag": tag,
        "induction_on": induction_on,
        "feature_on": feature_on,
        "start": start,
        "end": now_iso(),
        "memory_clear_returncode": rc_mem,
        "main_returncode": rc_main,
        "status": status,
        "digest_path": digest_path or "",
        "model_name": model_name,
        "run_timestamp": run_timestamp,
        "rows_extracted": len(rows),
        "output_tail": ((out_mem or "") + "\n" + (out_main or ""))[-4000:],
    }
    return rec, rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L2: Run 2×2 ablation with N repetitions (self-contained)"
    )
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of complete 2×2 repetitions")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Timeout per single condition-run (s). 0 = no limit")
    p.add_argument("--cooldown", type=float, default=2.0,
                   help="Cooldown seconds between consecutive runs")
    p.add_argument("--base-seed", type=int, default=42,
                   help="Base random seed")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Output directory for per-run artifacts")
    p.add_argument("--merged-csv", type=str,
                   default=os.path.join(ROOT, "data", "experiments",
                                        "l2_merged_longtable.csv"),
                   help="Path for merged longtable CSV")
    p.add_argument("--manifest-jsonl", type=str,
                   default=os.path.join(ROOT, "data", "experiments",
                                        "l2_manifest.jsonl"),
                   help="Path for run manifest JSONL")
    p.add_argument("--only-combo", type=str, default="",
                   help="Run only one combo label: A, B, C, or D (for debugging)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Filter combos if --only-combo given
    combos = COMBO_SPECS
    if args.only_combo:
        label_map = {c["label"][0]: c for c in COMBO_SPECS}  # "A" → spec
        key = args.only_combo.strip().upper()
        if key in label_map:
            combos = [label_map[key]]
            print(f"  *** Running ONLY combo {label_map[key]['label']} ***")
        else:
            print(f"ERROR: unknown combo '{args.only_combo}'. Choose A/B/C/D.")
            return 1

    total_runs = args.repeats * len(combos)
    print("=" * 70)
    print("L2 REPETITION RUNNER (self-contained)")
    print(f"  repeats={args.repeats}  combos={len(combos)}  "
          f"total_runs={total_runs}")
    print(f"  timeout={args.timeout}s  cooldown={args.cooldown}s  "
          f"base_seed={args.base_seed}")
    print(f"  out_dir={args.out_dir}")
    print("=" * 70)

    os.makedirs(args.out_dir, exist_ok=True)
    experiment_id_base = f"l2_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    run_idx = 0

    for rep in range(1, args.repeats + 1):
        seed = args.base_seed + rep * 137
        seed_everything(seed)

        for combo in combos:
            run_idx += 1
            experiment_id = f"{experiment_id_base}_rep{rep:02d}_{combo['tag']}"
            print(f"\n{'─' * 60}")
            print(f"[{run_idx}/{total_runs}] repeat={rep}/{args.repeats}  "
                  f"induction={'ON' if combo['induction'] else 'OFF'}  "
                  f"feature={'ON' if combo['feature'] else 'OFF'}  "
                  f"({combo['label']})  seed={seed}")
            print(f"{'─' * 60}")

            rec, rows = run_one_condition(
                experiment_id=experiment_id,
                induction_on=combo["induction"],
                feature_on=combo["feature"],
                combo_tag=combo["tag"],
                repeat_index=rep,
                timeout_s=args.timeout,
            )
            rec["seed"] = seed
            summary.append(rec)
            all_rows.extend(rows)

            # per-run log
            log_path = os.path.join(args.out_dir,
                                    f"rep{rep:02d}_{combo['tag']}.log")
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(rec.get("output_tail", ""))

            # per-condition CSV
            per_csv = os.path.join(args.out_dir,
                                   f"rep{rep:02d}_{combo['tag']}.csv")
            if rows:
                write_longtable(rows, per_csv)

            # manifest entry
            rec_for_jsonl = dict(rec)
            rec_for_jsonl.pop("output_tail", None)
            append_jsonl(args.manifest_jsonl, rec_for_jsonl)

            print(f"  -> status={rec['status']}  rows={len(rows)}  "
                  f"digest={rec.get('digest_path', '')}")

            if run_idx < total_runs:
                time.sleep(args.cooldown)

    # --- merge ---
    ok = sum(1 for s in summary if s["status"] == "ok")
    print(f"\n{'=' * 70}")
    print(f"Done: {ok}/{total_runs} successful")
    print(f"{'=' * 70}")

    if all_rows:
        merged_csv = args.merged_csv
        write_longtable(all_rows, merged_csv)
        print(f"Merged longtable: {merged_csv}  ({len(all_rows)} rows)")
    else:
        print("WARNING: zero rows collected – all runs may have failed")
        return 1

    append_jsonl(args.manifest_jsonl, {
        "type": "l2_summary",
        "total_runs": total_runs,
        "successful": ok,
        "merged_csv": merged_csv,
        "timestamp": now_iso(),
    })

    print(f"\nManifest: {args.manifest_jsonl}")
    print("Next step – statistical analysis:")
    print(f"  python experiments/l2_statistics.py --csv {merged_csv}")
    return 0 if ok == total_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
