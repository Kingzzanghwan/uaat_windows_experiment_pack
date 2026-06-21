import pandas as pd
from pathlib import Path

path = Path(r"runs\e2_mvtec_ad\coverage_matched_nearest.csv")
df = pd.read_csv(path)

print("columns:")
print(df.columns.tolist())

print("\n전체 행 수:", len(df))

# coverage gap 기준별 요약
for gap_limit in [0.005, 0.01, 0.02, 0.05]:
    sub = df[df["coverage_gap"] <= gap_limit].copy()

    print("\n" + "=" * 70)
    print(f"coverage_gap <= {gap_limit}")
    print("행 수:", len(sub))

    if len(sub) == 0:
        continue

    for baseline in ["fixed", "uncertainty_grid"]:
        b = sub[sub["baseline"] == baseline].copy()

        if len(b) == 0:
            continue

        win = (b["risk_delta_baseline_minus_uaat"] > 0).sum()
        lose = (b["risk_delta_baseline_minus_uaat"] < 0).sum()
        tie = (b["risk_delta_baseline_minus_uaat"] == 0).sum()

        print(f"\n[{baseline}]")
        print("비교 개수:", len(b))
        print("UAAT 승:", win)
        print("UAAT 패:", lose)
        print("동률:", tie)
        print("UAAT 승률:", round(win / len(b) * 100, 2), "%")
        print("평균 risk 감소율(%):", round(b["risk_reduction_pct"].mean(), 3))
        print("중앙값 risk 감소율(%):", round(b["risk_reduction_pct"].median(), 3))
        print("최소 risk 감소율(%):", round(b["risk_reduction_pct"].min(), 3))
        print("최대 risk 감소율(%):", round(b["risk_reduction_pct"].max(), 3))
        print("평균 wrong_auto_delta:", round(b["wrong_auto_delta"].mean(), 6))

# 논문 표 후보: target 0.60, 0.70, 0.80, 0.90만 보기
targets = [0.60, 0.70, 0.80, 0.90]
paper = df[df["uaat_target"].round(2).isin(targets)].copy()

paper_cols = [
    "uaat_target",
    "baseline",
    "baseline_target",
    "uaat_coverage",
    "baseline_coverage",
    "coverage_gap",
    "uaat_risk",
    "baseline_risk",
    "risk_delta_baseline_minus_uaat",
    "risk_reduction_pct",
    "uaat_wrong_auto_rate",
    "baseline_wrong_auto_rate",
    "wrong_auto_delta",
    "n",
]

paper = paper[paper_cols]
out = Path(r"runs\e2_mvtec_ad\coverage_matched_paper_targets.csv")
paper.to_csv(out, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("논문용 target 0.60, 0.70, 0.80, 0.90 후보")
print(paper.to_string(index=False))
print("\n저장 완료:", out)