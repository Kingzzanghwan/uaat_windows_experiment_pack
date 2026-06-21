"""Diagnose calib vs test uncertainty (and score) distribution shift.

Why this exists
---------------
Under domain shift (e.g. iWildCam), the TEST split can have systematically higher
model uncertainty than the CALIB split. Any policy whose threshold rises with
uncertainty (like UAAT's monotone net) will then defer MORE on test than its
calib-fitted target coverage implies -- which makes coverage comparisons against
a score-only baseline (fixed) unfair.

This script just measures that shift so you can confirm the cause before changing
any modeling code. It uses only the feature CSV; no labels needed for the shift
itself.

Usage
-----
python experiments/analysis_uncertainty_shift.py --csv runs/e3_iwildcam/features.csv \
    --out analysis_out/e3/uncertainty_shift.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_col", default="split")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    for col in (a.split_col, "uncertainty", "score"):
        if col not in df.columns:
            raise SystemExit(f"CSV is missing required column: {col!r}")

    s = df[a.split_col].astype(str).str.lower()
    calib = df[s.eq("calib")]
    test = df[s.eq("test")]
    if len(calib) == 0 or len(test) == 0:
        raise SystemExit("CSV must contain both split='calib' and split='test' rows.")

    rows = []
    for metric in ("uncertainty", "score"):
        c = _summ(calib[metric].to_numpy())
        t = _summ(test[metric].to_numpy())
        rows.append({
            "metric": metric,
            "calib_mean": c["mean"], "test_mean": t["mean"],
            "test_minus_calib_mean": t["mean"] - c["mean"],
            "calib_p50": c["p50"], "test_p50": t["p50"],
            "calib_p90": c["p90"], "test_p90": t["p90"],
            "calib_std": c["std"], "test_std": t["std"],
            "calib_n": c["n"], "test_n": t["n"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")

    print("saved:", a.out)
    print(out.to_string(index=False))
    print()

    unc = out[out["metric"] == "uncertainty"].iloc[0]
    shift = float(unc["test_minus_calib_mean"])
    print("=" * 60)
    print(f"uncertainty shift (test_mean - calib_mean) = {shift:+.4f}")
    if shift > 0.02:
        print("=> TEST uncertainty is clearly HIGHER than CALIB.")
        print("   This confirms the suspected cause: UAAT's uncertainty-driven")
        print("   threshold rises on test, so it defers more than its calib target.")
        print("   The --recalibrate_on_test fix in train_eval_policy.py should help.")
    elif shift < -0.02:
        print("=> TEST uncertainty is LOWER than calib. The coverage gap likely has")
        print("   a different cause; share this output for further diagnosis.")
    else:
        print("=> Almost no uncertainty shift. The coverage gap is probably NOT driven")
        print("   by uncertainty distribution shift; share this output for further diagnosis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
