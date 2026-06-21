import re
from pathlib import Path

import pandas as pd


BASE = Path(r"runs\e2_mvtec_ad")
OUT_DIR = BASE / "category_worst_group_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_setting(folder_name: str):
    """
    실험 폴더 이름에서 cost ratio와 target coverage를 읽는다.
    지원:
    - policy
    - policy_cov60
    - policy_cov70
    - policy_cov80
    - policy_cov90
    - cost_2to1_cov60
    - cost_5to1_cov80
    - cost_10to1_cov90
    - cost_20to1_cov70
    """
    if folder_name == "policy":
        return {
            "setting_dir": folder_name,
            "setting_type": "main_policy",
            "cost_ratio": "10:1",
            "c_wrong": 10,
            "c_defer": 1,
            "target_coverage": 0.80,
        }

    m = re.match(r"policy_cov(\d+)$", folder_name)
    if m:
        cov = int(m.group(1)) / 100
        return {
            "setting_dir": folder_name,
            "setting_type": "coverage_sweep",
            "cost_ratio": "10:1",
            "c_wrong": 10,
            "c_defer": 1,
            "target_coverage": cov,
        }

    m = re.match(r"cost_(\d+)to1_cov(\d+)$", folder_name)
    if m:
        c_wrong = int(m.group(1))
        cov = int(m.group(2)) / 100
        return {
            "setting_dir": folder_name,
            "setting_type": "cost_sensitivity",
            "cost_ratio": f"{c_wrong}:1",
            "c_wrong": c_wrong,
            "c_defer": 1,
            "target_coverage": cov,
        }

    return None


def infer_policy(file_path: Path):
    name = file_path.stem
    return name.replace("group_metrics_", "")


def normalize_group_column(df: pd.DataFrame):
    """
    group_metrics 파일에서 category 컬럼 이름을 통일한다.
    """
    if "category" in df.columns:
        return df

    if "group" in df.columns:
        return df.rename(columns={"group": "category"})

    if "Unnamed: 0" in df.columns:
        return df.rename(columns={"Unnamed: 0": "category"})

    first_col = df.columns[0]
    return df.rename(columns={first_col: "category"})


def read_all_group_metrics():
    rows = []

    for folder in BASE.iterdir():
        if not folder.is_dir():
            continue

        setting = parse_setting(folder.name)
        if setting is None:
            continue

        for csv_path in folder.glob("group_metrics_*.csv"):
            policy = infer_policy(csv_path)

            df = pd.read_csv(csv_path)
            df = normalize_group_column(df)

            df.insert(0, "policy", policy)
            df.insert(0, "target_coverage", setting["target_coverage"])
            df.insert(0, "c_defer", setting["c_defer"])
            df.insert(0, "c_wrong", setting["c_wrong"])
            df.insert(0, "cost_ratio", setting["cost_ratio"])
            df.insert(0, "setting_type", setting["setting_type"])
            df.insert(0, "setting_dir", setting["setting_dir"])

            rows.append(df)

    if not rows:
        raise SystemExit(
            "group_metrics 파일을 찾지 못했습니다. "
            "runs\\e2_mvtec_ad\\policy_cov80 같은 폴더 안에 group_metrics_*.csv가 있는지 확인하세요."
        )

    out = pd.concat(rows, ignore_index=True)
    return out


def make_best_worst(all_df: pd.DataFrame):
    records = []

    group_cols = [
        "setting_dir",
        "setting_type",
        "cost_ratio",
        "c_wrong",
        "c_defer",
        "target_coverage",
        "policy",
    ]

    for key, g in all_df.groupby(group_cols):
        g = g.copy()

        worst = g.sort_values("risk", ascending=False).iloc[0]
        best = g.sort_values("risk", ascending=True).iloc[0]

        base = dict(zip(group_cols, key))

        records.append({
            **base,
            "type": "worst",
            "category": worst["category"],
            "coverage": worst.get("coverage", None),
            "risk": worst.get("risk", None),
            "wrong_auto_rate": worst.get("wrong_auto_rate", None),
            "auto_error_rate": worst.get("auto_error_rate", None),
            "defer_rate": worst.get("defer_rate", None),
            "n": worst.get("n", None),
        })

        records.append({
            **base,
            "type": "best",
            "category": best["category"],
            "coverage": best.get("coverage", None),
            "risk": best.get("risk", None),
            "wrong_auto_rate": best.get("wrong_auto_rate", None),
            "auto_error_rate": best.get("auto_error_rate", None),
            "defer_rate": best.get("defer_rate", None),
            "n": best.get("n", None),
        })

    return pd.DataFrame(records)


def make_delta_table(all_df: pd.DataFrame, baseline_policy: str):
    uaat = all_df[all_df["policy"] == "uaat_monotone"].copy()
    base = all_df[all_df["policy"] == baseline_policy].copy()

    keys = [
        "setting_dir",
        "setting_type",
        "cost_ratio",
        "c_wrong",
        "c_defer",
        "target_coverage",
        "category",
    ]

    merged = pd.merge(
        uaat,
        base,
        on=keys,
        suffixes=("_uaat", f"_{baseline_policy}"),
        how="inner",
    )

    if len(merged) == 0:
        return merged

    merged["baseline_policy"] = baseline_policy

    merged["risk_delta_baseline_minus_uaat"] = (
        merged[f"risk_{baseline_policy}"] - merged["risk_uaat"]
    )

    merged["risk_reduction_pct"] = (
        merged["risk_delta_baseline_minus_uaat"]
        / merged[f"risk_{baseline_policy}"]
        * 100
    )

    merged["wrong_auto_delta_baseline_minus_uaat"] = (
        merged[f"wrong_auto_rate_{baseline_policy}"]
        - merged["wrong_auto_rate_uaat"]
    )

    keep_cols = [
        "setting_dir",
        "setting_type",
        "cost_ratio",
        "c_wrong",
        "c_defer",
        "target_coverage",
        "category",
        "baseline_policy",
        "coverage_uaat",
        f"coverage_{baseline_policy}",
        "risk_uaat",
        f"risk_{baseline_policy}",
        "risk_delta_baseline_minus_uaat",
        "risk_reduction_pct",
        "wrong_auto_rate_uaat",
        f"wrong_auto_rate_{baseline_policy}",
        "wrong_auto_delta_baseline_minus_uaat",
        "auto_error_rate_uaat",
        f"auto_error_rate_{baseline_policy}",
        "defer_rate_uaat",
        f"defer_rate_{baseline_policy}",
        "n_uaat",
    ]

    existing_cols = [c for c in keep_cols if c in merged.columns]
    return merged[existing_cols]


def md_table(df: pd.DataFrame, max_rows=20):
    """
    tabulate 없이 간단한 markdown table 생성.
    """
    if df is None or len(df) == 0:
        return "_No rows._"

    show = df.head(max_rows).copy()
    cols = list(show.columns)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in show.iterrows():
        values = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def make_markdown_report(all_df, best_worst, delta_fixed, delta_uncertainty):
    report_path = OUT_DIR / "category_worst_group_report.md"

    main = all_df[
        (all_df["cost_ratio"] == "10:1")
        & (all_df["target_coverage"].round(2) == 0.80)
    ].copy()

    main_uaat = main[main["policy"] == "uaat_monotone"].copy()
    main_fixed = main[main["policy"] == "fixed"].copy()

    worst_uaat = main_uaat.sort_values("risk", ascending=False).head(10)
    best_uaat = main_uaat.sort_values("risk", ascending=True).head(10)

    if len(delta_fixed) > 0:
        main_delta = delta_fixed[
            (delta_fixed["cost_ratio"] == "10:1")
            & (delta_fixed["target_coverage"].round(2) == 0.80)
        ].copy()

        most_improved = main_delta.sort_values(
            "risk_delta_baseline_minus_uaat", ascending=False
        ).head(10)

        most_degraded = main_delta.sort_values(
            "risk_delta_baseline_minus_uaat", ascending=True
        ).head(10)
    else:
        most_improved = pd.DataFrame()
        most_degraded = pd.DataFrame()

    text = []
    text.append("# E2 MVTec AD Category Worst-Group Risk Analysis")
    text.append("")
    text.append("## 1. 목적")
    text.append("")
    text.append(
        "이 파일은 E2 MVTec AD 실험에서 category별 risk를 확인하기 위해 생성되었다. "
        "특히 어떤 category에서 UAAT가 잘 작동했는지, 어떤 category가 worst-group인지 확인하는 데 사용한다."
    )
    text.append("")
    text.append("## 2. 생성 파일")
    text.append("")
    text.append("- `category_group_metrics_all.csv`: 모든 setting과 policy의 category별 group metrics 통합 파일")
    text.append("- `category_best_worst_by_setting.csv`: setting별 best/worst category 요약")
    text.append("- `category_uaat_vs_fixed_deltas.csv`: category별 UAAT와 fixed 차이")
    text.append("- `category_uaat_vs_uncertainty_grid_deltas.csv`: category별 UAAT와 uncertainty_grid 차이")
    text.append("- `category_worst_group_report.md`: 현재 요약 보고서")
    text.append("")
    text.append("## 3. Main setting: cost 10:1, target coverage 0.80")
    text.append("")
    text.append("### 3.1 UAAT worst-risk categories")
    text.append("")
    text.append(md_table(worst_uaat[[
        "category", "coverage", "risk", "wrong_auto_rate",
        "auto_error_rate", "defer_rate", "n"
    ]], 10))
    text.append("")
    text.append("### 3.2 UAAT best-risk categories")
    text.append("")
    text.append(md_table(best_uaat[[
        "category", "coverage", "risk", "wrong_auto_rate",
        "auto_error_rate", "defer_rate", "n"
    ]], 10))
    text.append("")
    text.append("### 3.3 UAAT improvement over fixed")
    text.append("")
    text.append("`risk_delta_baseline_minus_uaat`가 양수이면 UAAT가 fixed보다 risk를 낮춘 것이다.")
    text.append("")
    if len(most_improved) > 0:
        text.append(md_table(most_improved[[
            "category",
            "coverage_uaat",
            "coverage_fixed",
            "risk_uaat",
            "risk_fixed",
            "risk_delta_baseline_minus_uaat",
            "risk_reduction_pct",
            "wrong_auto_delta_baseline_minus_uaat",
        ]], 10))
    else:
        text.append("_No rows._")
    text.append("")
    text.append("### 3.4 Categories where UAAT is weakest against fixed")
    text.append("")
    text.append("음수 값이 있으면 해당 category에서는 fixed가 UAAT보다 risk가 낮다는 뜻이다.")
    text.append("")
    if len(most_degraded) > 0:
        text.append(md_table(most_degraded[[
            "category",
            "coverage_uaat",
            "coverage_fixed",
            "risk_uaat",
            "risk_fixed",
            "risk_delta_baseline_minus_uaat",
            "risk_reduction_pct",
            "wrong_auto_delta_baseline_minus_uaat",
        ]], 10))
    else:
        text.append("_No rows._")
    text.append("")
    text.append("## 4. 해석 방법")
    text.append("")
    text.append("- `risk`가 높은 category는 worst-group으로 볼 수 있다.")
    text.append("- `risk_delta_baseline_minus_uaat > 0`이면 UAAT가 baseline보다 좋다.")
    text.append("- `risk_delta_baseline_minus_uaat < 0`이면 해당 category에서는 UAAT가 baseline보다 나쁘다.")
    text.append("- `wrong_auto_delta_baseline_minus_uaat > 0`이면 UAAT가 잘못된 자동 판단을 줄인 것이다.")
    text.append("")
    text.append("## 5. 논문에서 사용할 수 있는 설명")
    text.append("")
    text.append(
        "전체 평균 결과뿐 아니라 category별 worst-group risk를 추가로 분석하였다. "
        "이를 통해 UAAT가 특정 category에서만 효과적인지, 또는 여러 category에서 일관되게 risk를 줄이는지 확인할 수 있다. "
        "또한 worst category를 별도로 보고함으로써 평균 성능 뒤에 가려진 실패 사례를 함께 제시할 수 있다."
    )
    text.append("")

    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def main():
    print("Reading group metrics from:", BASE)

    all_df = read_all_group_metrics()

    all_path = OUT_DIR / "category_group_metrics_all.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    best_worst = make_best_worst(all_df)
    best_worst_path = OUT_DIR / "category_best_worst_by_setting.csv"
    best_worst.to_csv(best_worst_path, index=False, encoding="utf-8-sig")

    delta_fixed = make_delta_table(all_df, "fixed")
    delta_fixed_path = OUT_DIR / "category_uaat_vs_fixed_deltas.csv"
    delta_fixed.to_csv(delta_fixed_path, index=False, encoding="utf-8-sig")

    delta_uncertainty = make_delta_table(all_df, "uncertainty_grid")
    delta_uncertainty_path = OUT_DIR / "category_uaat_vs_uncertainty_grid_deltas.csv"
    delta_uncertainty.to_csv(delta_uncertainty_path, index=False, encoding="utf-8-sig")

    report_path = make_markdown_report(
        all_df,
        best_worst,
        delta_fixed,
        delta_uncertainty,
    )

    print("\nSaved files:")
    print(all_path)
    print(best_worst_path)
    print(delta_fixed_path)
    print(delta_uncertainty_path)
    print(report_path)

    print("\nDone.")


if __name__ == "__main__":
    main()