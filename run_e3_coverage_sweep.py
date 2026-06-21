import subprocess
import sys
from pathlib import Path
import pandas as pd

base = Path(r"runs\e3_iwildcam")
features_csv = base / "features.csv"

coverages = [0.60, 0.70, 0.80, 0.90]
rows = []

for cov in coverages:
    cov_tag = int(round(cov * 100))
    out_dir = base / f"policy_cov{cov_tag}"
    metrics_path = out_dir / "metrics.csv"

    if not metrics_path.exists():
        print(f"\n=== Running E3 iWildCam coverage {cov:.2f}, cost 5:1 ===")
        cmd = [
            sys.executable,
            r"experiments\train_eval_policy.py",
            "--csv", str(features_csv),
            "--out_dir", str(out_dir),
            "--target_coverage", f"{cov:.2f}",
            "--c_wrong", "5",
            "--c_defer", "1",
            "--group_col", "wilds_split",
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"Already exists: {metrics_path}")

    df = pd.read_csv(metrics_path)
    df.insert(0, "target_coverage", cov)
    df.insert(1, "cost_ratio", "5:1")
    rows.append(df)

out = pd.concat(rows, ignore_index=True)

out_path = base / "coverage_comparison.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\nSaved:", out_path)
print(out.to_string(index=False))