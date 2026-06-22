"""Group / worst-case risk analysis that actually splits by domain.

Why this replaces the old path
------------------------------
The previous group analysis grouped by the column 'wilds_split', but in the
feature extraction code that column only stores the WILDS *split name*
(literally the string "test"). Grouping the test set by it yields exactly ONE
group, so worst-case-by-domain is meaningless (n_groups=1).

The real per-image domain signal (camera-trap location etc.) is stored in the
extracted context columns, e.g. 'ctx_meta_location', and these ARE present in
each run's test_decisions.csv (which is a full copy of the test rows plus each
policy's accept flag). This script reads test_decisions.csv directly, bins a
chosen domain column into groups, and computes per-policy mean / worst-group
risk. No GPU re-extraction is required.

Usage
-----
python experiments/analysis_group_worst_fixed.py \
    --decisions runs/e3_iwildcam/policy_cov80/test_decisions.csv \
    --group_col ctx_meta_location --n_bins 10 \
    --c_wrong 5 --c_defer 1 \
    --out analysis_out/e3/group_worst_cov80_fixed.csv

If you are unsure which column to use, run with --list_cols to print candidate
domain columns and exit.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _risk(accept: np.ndarray, error: np.ndarray, c_wrong: float, c_defer: float) -> dict:
    accept = accept.astype(int)
    error = error.astype(int)
    n = len(accept)
    coverage = float(accept.mean()) if n else 0.0
    wrong_auto = float(((accept == 1) & (error == 1)).mean()) if n else 0.0
    risk = c_wrong * wrong_auto + c_defer * (1.0 - coverage)
    return {"coverage": coverage, "wrong_auto_rate": wrong_auto, "risk": risk, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help="A test_decisions.csv from train_eval_policy.py")
    ap.add_argument("--group_col", default="ctx_meta_location",
                    help="Column to define domain groups (default ctx_meta_location)")
    ap.add_argument("--n_bins", type=int, default=10,
                    help="If the group column is continuous/has many values, bin into this many quantile buckets")
    ap.add_argument("--min_group_n", type=int, default=50,
                    help="Ignore groups smaller than this many samples (too few to be meaningful)")
    ap.add_argument("--c_wrong", type=float, default=5.0)
    ap.add_argument("--c_defer", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list_cols", action="store_true", help="List candidate domain columns and exit")
    a = ap.parse_args()

    df = pd.read_csv(a.decisions)

    if a.list_cols:
        cand = [c for c in df.columns if c.startswith("ctx") or "location" in c.lower()
                or "domain" in c.lower() or "group" in c.lower()]
        print("Candidate domain/group columns and their #unique values:")
        for c in cand:
            print(f"  {c:30s} unique={df[c].nunique()}")
        if not cand:
            print("  (none found - check the CSV header)")
        return

    accept_cols = [c for c in df.columns if c.startswith("accept_")]
    if not accept_cols:
        raise SystemExit("No accept_<policy> columns found - is this a test_decisions.csv?")
    if "error" not in df.columns:
        raise SystemExit("No 'error' column found in the decisions CSV.")
    if a.group_col not in df.columns:
        cand = [c for c in df.columns if c.startswith("ctx") or "location" in c.lower()]
        raise SystemExit(
            f"group column {a.group_col!r} not in CSV. "
            f"Available candidate columns: {cand}. "
            f"Re-run with --group_col <one of these>, or --list_cols to inspect."
        )

    # Build the grouping key. If the column has many distinct values or is
    # continuous (normalized location ids are floats in [0,1]), bucket it into
    # quantile bins so each group has a reasonable sample size.
    col = df[a.group_col]
    nuniq = col.nunique()
    if nuniq > a.n_bins and pd.api.types.is_numeric_dtype(col):
        # quantile binning; drop duplicate edges gracefully
        try:
            df["_group_key"] = pd.qcut(col, q=a.n_bins, duplicates="drop")
        except Exception:
            df["_group_key"] = pd.cut(col, bins=a.n_bins)
        group_kind = f"{a.n_bins}-quantile-bins of {a.group_col}"
    else:
        df["_group_key"] = col.astype(str)
        group_kind = f"raw values of {a.group_col}"

    rows = []
    for name in accept_cols:
        policy = name[len("accept_"):]
        per_group = []
        for gkey, g in df.groupby("_group_key", observed=True):
            if len(g) < a.min_group_n:
                continue
            r = _risk(g[name].to_numpy(), g["error"].to_numpy(), a.c_wrong, a.c_defer)
            r["group"] = str(gkey)
            per_group.append(r)
        if not per_group:
            continue
        gdf = pd.DataFrame(per_group)
        worst_idx = gdf["risk"].idxmax()
        rows.append({
            "policy": policy,
            "n_groups": int(len(gdf)),
            "mean_group_risk": float(gdf["risk"].mean()),
            "median_group_risk": float(gdf["risk"].median()),
            "worst_group_risk": float(gdf.loc[worst_idx, "risk"]),
            "worst_group": str(gdf.loc[worst_idx, "group"]),
            "best_group_risk": float(gdf["risk"].min()),
            "group_kind": group_kind,
        })

    if not rows:
        raise SystemExit(
            "No groups passed the --min_group_n filter. Try a smaller --n_bins "
            "or --min_group_n, or a different --group_col."
        )

    out = pd.DataFrame(rows).sort_values("worst_group_risk")
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print("saved:", a.out)
    print(f"grouping: {group_kind}")
    print(out.to_string(index=False))
    print()
    # quick verdict
    if {"uaat_monotone", "fixed"}.issubset(set(out["policy"])):
        u = out[out.policy == "uaat_monotone"].iloc[0]
        f = out[out.policy == "fixed"].iloc[0]
        better = u["worst_group_risk"] < f["worst_group_risk"]
        print(f"UAAT worst-group risk {u['worst_group_risk']:.4f} vs fixed {f['worst_group_risk']:.4f}"
              f"  ->  {'UAAT better (more robust on worst domain)' if better else 'fixed better on worst domain'}")


if __name__ == "__main__":
    main()
