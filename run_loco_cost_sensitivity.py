import subprocess
import sys
from pathlib import Path
import pandas as pd

base = Path(r"runs\e2_mvtec_loco")
features_csv = base / "features.csv"

costs = [2, 5, 10, 20]
coverages = [0.60, 0.70, 0.80, 0.90]

rows = []

for c_wrong in costs:
    for cov in coverages:
        cov_tag = int(round(cov * 100))
        out_dir = base / f"cost_{c_wrong}to1_cov{cov_tag}"
        metrics_path = out_dir / "metrics.csv"

        if not metrics_path.exists():
            print(f"\n=== Running LOCO cost {c_wrong}:1, target coverage {cov:.2f} ===")
            cmd = [
                sys.executable,
                r"experiments\train_eval_policy.py",
                "--csv", str(features_csv),
                "--out_dir", str(out_dir),
                "--target_coverage", f"{cov:.2f}",
                "--c_wrong", str(c_wrong),
                "--c_defer", "1",
                "--group_col", "category",
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"Already exists: {metrics_path}")

        df = pd.read_csv(metrics_path)
        df.insert(0, "cost_ratio", f"{c_wrong}:1")
        df.insert(1, "c_wrong", c_wrong)
        df.insert(2, "c_defer", 1)
        df.insert(3, "target_coverage", cov)
        rows.append(df)

out = pd.concat(rows, ignore_index=True)

out_path = base / "cost_sensitivity_comparison.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\nSaved:", out_path)
print(out.to_string(index=False))