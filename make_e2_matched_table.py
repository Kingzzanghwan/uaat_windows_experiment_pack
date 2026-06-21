import subprocess
import sys
from pathlib import Path

import pandas as pd


base = Path(r"runs\e2_mvtec_ad")
features_csv = base / "features.csv"

targets = [i / 100 for i in range(40, 96)]  # 0.40 ~ 0.95

all_rows = []

for target in targets:
    tag = int(round(target * 100))
    out_dir = base / f"policy_grid_{tag:02d}"
    metrics_path = out_dir / "metrics.csv"

    if not metrics_path.exists():
        print(f"\n=== Running target_coverage={target:.2f} ===")
        cmd = [
            sys.executable,
            r"experiments\train_eval_policy.py",
            "--csv", str(features_csv),
            "--out_dir", str(out_dir),
            "--target_coverage", f"{target:.2f}",
            "--c_wrong", "10",
            "--c_defer", "1",
            "--group_col", "category",
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"Already exists: {metrics_path}")

    df = pd.read_csv(metrics_path)
    df.insert(0, "target_coverage", target)
    df.insert(1, "result_dir", str(out_dir))
    all_rows.append(df)

grid = pd.concat(all_rows, ignore_index=True)

grid_path = base / "coverage_grid_all.csv"
grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
print("\nSaved:", grid_path)


# coverage-matched nearest comparison
uaat = grid[grid["policy"] == "uaat_monotone"].copy()
baselines = ["fixed", "uncertainty_grid"]

matched_rows = []

for _, u in uaat.iterrows():
    for baseline in baselines:
        cand = grid[grid["policy"] == baseline].copy()
        cand["coverage_gap"] = (cand["coverage"] - u["coverage"]).abs()
        b = cand.sort_values("coverage_gap").iloc[0]

        matched_rows.append({
            "uaat_target": u["target_coverage"],
            "baseline": baseline,
            "baseline_target": b["target_coverage"],

            "uaat_coverage": u["coverage"],
            "baseline_coverage": b["coverage"],
            "coverage_gap": b["coverage_gap"],

            "uaat_risk": u["risk"],
            "baseline_risk": b["risk"],
            "risk_delta_baseline_minus_uaat": b["risk"] - u["risk"],
            "risk_reduction_pct": (b["risk"] - u["risk"]) / b["risk"] * 100 if b["risk"] != 0 else 0,

            "uaat_wrong_auto_rate": u["wrong_auto_rate"],
            "baseline_wrong_auto_rate": b["wrong_auto_rate"],
            "wrong_auto_delta": b["wrong_auto_rate"] - u["wrong_auto_rate"],

            "uaat_auto_error_rate": u["auto_error_rate"],
            "baseline_auto_error_rate": b["auto_error_rate"],

            "uaat_defer_rate": u["defer_rate"],
            "baseline_defer_rate": b["defer_rate"],

            "n": u["n"],
        })

matched = pd.DataFrame(matched_rows)

matched_path = base / "coverage_matched_nearest.csv"
matched.to_csv(matched_path, index=False, encoding="utf-8-sig")

print("\nSaved:", matched_path)
print("\n=== Coverage-matched nearest comparison ===")
print(matched.to_string(index=False))