from pathlib import Path
import pandas as pd
from seed3_utils import project_root

project_root()

EXPERIMENTS = [
    "e1_cifar10c_all",
    "e2_mvtec_ad",
    "e2_mvtec_loco",
    "e3_iwildcam",
]
TABLES = [
    "coverage_comparison.csv",
    "coverage_matched_paper_targets.csv",
    "cost_sensitivity_comparison.csv",
]

summary_dir = Path(r"runs_seed3\_summary")
summary_dir.mkdir(parents=True, exist_ok=True)


def summarize_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in ["seed"] and c not in group_cols]
    grouped = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in grouped.columns]
    return grouped

for exp in EXPERIMENTS:
    exp_dir = Path("runs_seed3") / exp
    for table in TABLES:
        parts = []
        for seed_dir in sorted(exp_dir.glob("seed_*")):
            p = seed_dir / table
            if p.exists():
                parts.append(pd.read_csv(p))
        if not parts:
            print(f"Skip: {exp}/{table}, no files")
            continue
        df = pd.concat(parts, ignore_index=True)

        raw_path = summary_dir / f"{exp}__{table.replace('.csv', '')}__all_seeds.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print("Saved raw:", raw_path)

        if table == "coverage_comparison.csv":
            group_cols = ["target_coverage", "cost_ratio", "policy"]
        elif table == "cost_sensitivity_comparison.csv":
            group_cols = ["cost_ratio", "c_wrong", "c_defer", "target_coverage", "policy"]
        elif table == "coverage_matched_paper_targets.csv":
            group_cols = ["uaat_target", "baseline", "baseline_target", "cost_ratio"]
        else:
            group_cols = []

        if group_cols:
            existing_group_cols = [c for c in group_cols if c in df.columns]
            summary = summarize_numeric(df, existing_group_cols)
            sum_path = summary_dir / f"{exp}__{table.replace('.csv', '')}__mean_std.csv"
            summary.to_csv(sum_path, index=False, encoding="utf-8-sig")
            print("Saved summary:", sum_path)

print("\n완료. 요약 파일 위치:", summary_dir)
