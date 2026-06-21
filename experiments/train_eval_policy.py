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
    return p.parse_args()


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

    fixed = fixed_threshold_policy(calib, args.target_coverage)
    policies.append(("fixed", fixed, apply_fixed(test, fixed), fixed))

    ug = uncertainty_grid_policy(
        calib,
        target_coverage=args.target_coverage,
        c_wrong=args.c_wrong,
        c_defer=args.c_defer,
    )
    policies.append(("uncertainty_grid", ug, apply_uncertainty_grid(test, ug), ug))

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
    policies.append(("uaat_monotone", net, test["score"].to_numpy() > tau, net.to_dict()))

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
        },
        out_dir / "run_config.json",
    )
    print("\n=== UAAT policy results ===")
    print(results.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
