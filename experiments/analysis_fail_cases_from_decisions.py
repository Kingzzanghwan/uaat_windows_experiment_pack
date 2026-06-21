"""Extract UAAT failure cases from test_decisions.csv.

The training driver (train_eval_policy.py) writes test_decisions.csv per run,
containing per-sample columns: score, uncertainty, error, accept_<policy>, and
tau_uaat_monotone. This script ranks the most informative failures where UAAT
accepted (auto-decided) but was wrong, plus a comparison of where UAAT and the
fixed baseline disagree. No image paths are required; if a path column exists
(e.g. 'path' or 'filepath'), it is carried through so you can pull images later.

Usage:
  python experiments/analysis_fail_cases_from_decisions.py \
      --csv runs/e3_iwildcam/policy_cov80/test_decisions.csv \
      --out analysis_out/e3/fail_cases_cov80.csv --k 20
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="A test_decisions.csv produced by train_eval_policy.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20, help="How many worst cases to keep")
    ap.add_argument("--policy", default="uaat_monotone")
    ap.add_argument("--base", default="fixed")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    accept_col = f"accept_{a.policy}"
    base_col = f"accept_{a.base}"
    needed = {"score", "uncertainty", "error", accept_col}
    miss = needed - set(df.columns)
    if miss:
        raise SystemExit(f"CSV is missing required columns: {sorted(miss)}")

    df = df.copy()
    df["error"] = df["error"].astype(int)
    df[accept_col] = df[accept_col].astype(int)
    if base_col in df.columns:
        df[base_col] = df[base_col].astype(int)

    # carry through any obvious id/path column for later image pulls
    path_col = next((c for c in ("path", "filepath", "image", "img_path", "filename") if c in df.columns), None)
    keep_cols = ["score", "uncertainty", "error", accept_col]
    if base_col in df.columns:
        keep_cols.append(base_col)
    if "tau_uaat_monotone" in df.columns:
        keep_cols.append("tau_uaat_monotone")
    if path_col:
        keep_cols.insert(0, path_col)

    # The dangerous failures: UAAT auto-decided (accept=1) but was wrong (error=1).
    # Rank by lowest "margin of safety": for UAAT, score - tau (how barely it
    # passed). If tau column is present use it; otherwise fall back to -score
    # (most confident-but-wrong first is also informative, so we sort by score asc).
    wrong_auto = df[(df[accept_col] == 1) & (df["error"] == 1)].copy()
    if "tau_uaat_monotone" in wrong_auto.columns:
        wrong_auto["safety_margin"] = wrong_auto["score"] - wrong_auto["tau_uaat_monotone"]
        wrong_auto = wrong_auto.sort_values("safety_margin")  # smallest margin = barely accepted
    else:
        wrong_auto["safety_margin"] = np.nan
        wrong_auto = wrong_auto.sort_values("score")  # low score but accepted

    worst = wrong_auto[keep_cols + (["safety_margin"] if "safety_margin" in wrong_auto.columns else [])].head(a.k)
    worst.to_csv(a.out, index=False, encoding="utf-8-sig")

    # quick summary
    n = len(df)
    n_wrong_auto = int(((df[accept_col] == 1) & (df["error"] == 1)).sum())
    print("saved:", a.out)
    print(f"total test samples: {n}")
    print(f"UAAT wrong-auto (accepted but error): {n_wrong_auto}  ({100.0*n_wrong_auto/max(1,n):.2f}%)")
    if base_col in df.columns:
        only_uaat_wrong = int(((df[accept_col] == 1) & (df[base_col] == 0) & (df["error"] == 1)).sum())
        only_base_wrong = int(((df[base_col] == 1) & (df[accept_col] == 0) & (df["error"] == 1)).sum())
        print(f"wrong-auto only UAAT made (base deferred): {only_uaat_wrong}")
        print(f"wrong-auto only {a.base} made (UAAT deferred): {only_base_wrong}")
        print("  -> if 'only UAAT' < 'only base', UAAT avoided more dangerous auto-errors.")
    print()
    print(f"top {min(a.k, len(worst))} worst cases written. Columns:", list(worst.columns))


if __name__ == "__main__":
    main()
