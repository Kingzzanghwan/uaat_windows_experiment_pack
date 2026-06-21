import pandas as pd
from pathlib import Path

base = Path(r"runs\e2_mvtec_ad")

rows = []

for cov in [60, 70, 80, 90]:
    metrics_path = base / f"policy_cov{cov}" / "metrics.csv"

    if not metrics_path.exists():
        print(f"없음: {metrics_path}")
        continue

    df = pd.read_csv(metrics_path)
    df.insert(0, "target_coverage", cov / 100)
    rows.append(df)

if not rows:
    raise SystemExit("합칠 metrics.csv가 없습니다.")

out = pd.concat(rows, ignore_index=True)

out_path = base / "coverage_comparison.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")

print("저장 완료:", out_path)
print(out.to_string(index=False))