from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uaat.metrics import (  # noqa: E402
    apply_fixed,
    apply_uncertainty_grid,
    decision_metrics,
    fixed_threshold_policy,
    uncertainty_grid_policy,
)
from uaat.policy import PolicyTrainConfig, train_monotone_policy_net  # noqa: E402
from uaat.utils import ensure_dir, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate UAAT decision policies from feature CSV.")
    p.add_argument("--csv", required=True, help="CSV with score,error,uncertainty,split,ctx_* columns")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_coverage", type=float, default=0.80)
    p.add_argument("--c_wrong", type=float, default=5.0)
    p.add_argument("--c_defer", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy_steps", type=int, default=1200)
    p.add_argument("--context_prefix", default="ctx_")
    p.add_argument("--group_col", default="", help="Optional column for per-group summary, e.g. corruption or category")
    p.add_argument(
        "--recalibrate_on_test",
        action="store_true",
        help=(
            "Coverage-calibrate every policy on the TEST inputs so each one actually "
            "achieves --target_coverage on test (labels are NOT used; only the score / "
            "decision-margin distribution is used to pick the cutoff). This removes the "
            "calib->test coverage drift that otherwise makes coverage comparisons unfair "
            "under domain shift. When this flag is omitted, behavior is identical to before."
        ),
    )
    return p.parse_args()


def _quantile_cutoff_for_coverage(margin: np.ndarray, target_coverage: float) -> float:
    """Return the cutoff c such that (margin > c) keeps ~target_coverage fraction.

    Uses only the margin values (no labels), so this is a coverage calibration,
    not a risk calibration -> no test-label leakage.
    """
    margin = np.asarray(margin, dtype=np.float64)
    target_coverage = float(min(max(target_coverage, 1e-6), 1.0 - 1e-6))
    return float(np.quantile(margin, 1.0 - target_coverage))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)
    required = {"score", "error", "uncertainty", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["score", "error", "uncertainty", "split"]).reset_index(drop=True)
    df["score"] = df["score"].clip(0, 1)
    df["uncertainty"] = df["uncertainty"].clip(0, 1)
    df["error"] = df["error"].astype(int)

    calib = df[df["split"].astype(str).str.lower().eq("calib")].copy()
    test = df[df["split"].astype(str).str.lower().eq("test")].copy()
    if len(calib) == 0 or len(test) == 0:
        raise ValueError("CSV must contain both split='calib' and split='test' rows.")

    context_cols = [c for c in df.columns if c.startswith(args.context_prefix)]
    # Avoid accidental string context columns.
    numeric_context_cols = []
    for c in context_cols:
        try:
            calib[c] = pd.to_numeric(calib[c], errors="coerce").fillna(0.0)
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0.0)
            numeric_context_cols.append(c)
        except Exception:
            pass
    context_cols = numeric_context_cols

    policies: list[tuple[str, object, np.ndarray, dict]] = []

    # For each policy we keep the per-sample decision MARGIN on test
    # (margin > cutoff  <=>  accept). When --recalibrate_on_test is set we
    # re-pick each cutoff from the test margin distribution so every policy
    # hits the same target coverage on test. Labels are never used here.
    fixed = fixed_threshold_policy(calib, args.target_coverage)
    fixed_margin = test["score"].to_numpy(dtype=np.float64)
    if args.recalibrate_on_test:
        fixed_cut = _quantile_cutoff_for_coverage(fixed_margin, args.target_coverage)
        fixed = {**fixed, "threshold": fixed_cut, "recalibrated_on_test": True}
        fixed_accept = fixed_margin > fixed_cut
    else:
        fixed_accept = apply_fixed(test, fixed)
    policies.append(("fixed", fixed, fixed_accept, fixed))

    ug = uncertainty_grid_policy(
        calib,
        target_coverage=args.target_coverage,
        c_wrong=args.c_wrong,
        c_defer=args.c_defer,
    )
    ug_margin = (
        test["score"].to_numpy(dtype=np.float64)
        - float(ug["alpha"]) * test["uncertainty"].to_numpy(dtype=np.float64)
    )
    if args.recalibrate_on_test:
        ug_cut = _quantile_cutoff_for_coverage(ug_margin, args.target_coverage)
        ug = {**ug, "theta": ug_cut, "recalibrated_on_test": True}
        ug_accept = ug_margin > ug_cut
    else:
        ug_accept = apply_uncertainty_grid(test, ug)
    policies.append(("uncertainty_grid", ug, ug_accept, ug))

    net = train_monotone_policy_net(
        calib,
        context_cols=context_cols,
        cfg=PolicyTrainConfig(
            target_coverage=args.target_coverage,
            c_wrong=args.c_wrong,
            c_defer=args.c_defer,
            steps=args.policy_steps,
            seed=args.seed,
        ),
    )
    tau = net.predict_tau(test)
    # UAAT decision margin = score - tau(x). Accept if score > tau(x), i.e. margin > 0.
    uaat_margin = test["score"].to_numpy(dtype=np.float64) - np.asarray(tau, dtype=np.float64)
    uaat_dict = net.to_dict()
    if args.recalibrate_on_test:
        # Shift the (already-shaped) UAAT threshold by a single global offset so
        # that test coverage matches target. This preserves tau(x)'s dependence
        # on uncertainty/context (the learned shape) and only moves its height.
        uaat_cut = _quantile_cutoff_for_coverage(uaat_margin, args.target_coverage)
        uaat_accept = uaat_margin > uaat_cut
        uaat_dict = {**uaat_dict, "test_margin_offset": uaat_cut, "recalibrated_on_test": True}
    else:
        uaat_accept = test["score"].to_numpy() > tau
    policies.append(("uaat_monotone", net, uaat_accept, uaat_dict))

    rows = []
    decisions = test.copy()
    for name, _policy_obj, accept, policy_dict in policies:
        m = decision_metrics(test, accept, c_wrong=args.c_wrong, c_defer=args.c_defer).as_dict()
        rows.append({"policy": name, **m})
        decisions[f"accept_{name}"] = accept.astype(int)
        if name == "uaat_monotone":
            decisions["tau_uaat_monotone"] = tau

        if args.group_col and args.group_col in test.columns:
            group_rows = []
            for group, g in test.groupby(args.group_col):
                idx = g.index.to_numpy()
                local_accept = pd.Series(accept, index=test.index).loc[idx].to_numpy()
                gm = decision_metrics(g, local_accept, c_wrong=args.c_wrong, c_defer=args.c_defer).as_dict()
                group_rows.append({"policy": name, args.group_col: group, **gm})
            pd.DataFrame(group_rows).to_csv(out_dir / f"group_metrics_{name}.csv", index=False)

        with (out_dir / f"policy_{name}.json").open("w", encoding="utf-8") as f:
            json.dump(policy_dict, f, indent=2, ensure_ascii=False)

    results = pd.DataFrame(rows).sort_values("risk")
    results.to_csv(out_dir / "metrics.csv", index=False)
    decisions.to_csv(out_dir / "test_decisions.csv", index=False)

    save_json(
        {
            "csv": str(args.csv),
            "n_calib": int(len(calib)),
            "n_test": int(len(test)),
            "target_coverage": args.target_coverage,
            "c_wrong": args.c_wrong,
            "c_defer": args.c_defer,
            "context_cols": context_cols,
            "recalibrate_on_test": bool(args.recalibrate_on_test),
        },
        out_dir / "run_config.json",
    )
    print("\n=== UAAT policy results ===")
    print(results.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
