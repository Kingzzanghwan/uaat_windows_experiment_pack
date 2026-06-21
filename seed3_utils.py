from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from functools import lru_cache
import pandas as pd

SEEDS = [1, 2, 3]
COVERAGES = [0.60, 0.70, 0.80, 0.90]
GRID_TARGETS = [i / 100 for i in range(40, 96)]
COSTS = [2, 5, 10, 20]
PAPER_TARGETS = [0.60, 0.70, 0.80, 0.90]


def project_root() -> Path:
    root = Path.cwd()
    if not (root / "experiments").exists():
        raise SystemExit(
            "ERROR: 이 스크립트는 C:\\UAAT\\uaat_windows_experiment_pack 위치에서 실행해야 합니다.\n"
            "먼저 실행하세요:\n"
            "  cd /d C:\\UAAT\\uaat_windows_experiment_pack\n"
            "  .venv\\Scripts\\activate\n"
            "  set PYTHONPATH=%CD%\\src"
        )
    return root


def path_s(p: Path | str) -> str:
    return str(p)


@lru_cache(maxsize=None)
def help_text(script: str) -> str:
    cmd = [sys.executable, script, "--help"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        return ""


def supports_arg(script: str, arg: str) -> bool:
    return arg in help_text(script)


def add_if_supported(cmd: list[str], script: str, arg: str, value: str | None = None) -> list[str]:
    if supports_arg(script, arg):
        cmd.append(arg)
        if value is not None:
            cmd.append(value)
    return cmd


def add_seed_if_supported(cmd: list[str], script: str, seed: int) -> list[str]:
    # 실험 스크립트가 --seed를 지원할 때만 붙인다.
    # 지원하지 않으면 환경변수만 설정된다.
    return add_if_supported(cmd, script, "--seed", str(seed))


def run_cmd(cmd: list[str], seed: int | None = None) -> None:
    env = os.environ.copy()
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
        env["UAAT_SEED"] = str(seed)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def find_existing(candidates: list[str]) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError("다음 후보 경로를 찾지 못했습니다: " + ", ".join(candidates))


def detect_group_col(features_csv: Path, preferred: str | None = None) -> str:
    df = pd.read_csv(features_csv, nrows=5)
    cols = set(df.columns)
    if preferred and preferred in cols:
        return preferred
    for c in ["wilds_split", "category", "corruption", "corruption_name", "severity", "split"]:
        if c in cols:
            return c
    raise ValueError(f"group_col을 자동으로 찾지 못했습니다. CSV columns={list(df.columns)}")


def require_features(features_csv: Path) -> None:
    if not features_csv.exists():
        raise FileNotFoundError(f"features.csv가 없습니다: {features_csv}")
    print(f"OK features: {features_csv}")


def read_metrics(metrics_path: Path, seed: int, extra: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    df.insert(0, "seed", seed)
    if extra:
        for idx, (k, v) in enumerate(extra.items(), start=1):
            df.insert(idx, k, v)
    return df


def run_train_eval_policy(features_csv: Path, out_dir: Path, target_coverage: float, c_wrong: int, c_defer: int, group_col: str) -> Path:
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        print(f"Already exists: {metrics_path}")
        return metrics_path
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        r"experiments\train_eval_policy.py",
        "--csv", path_s(features_csv),
        "--out_dir", path_s(out_dir),
        "--target_coverage", f"{target_coverage:.2f}",
        "--c_wrong", str(c_wrong),
        "--c_defer", str(c_defer),
        "--group_col", group_col,
    ]
    run_cmd(cmd)
    return metrics_path


def run_policy_suite(base: Path, features_csv: Path, seed: int, group_col: str, main_c_wrong: int, c_defer: int = 1) -> None:
    base.mkdir(parents=True, exist_ok=True)
    require_features(features_csv)
    print(f"\n=== Policy suite: base={base}, seed={seed}, group_col={group_col}, main_cost={main_c_wrong}:1 ===")

    # 1) coverage comparison
    coverage_rows = []
    for cov in COVERAGES:
        cov_tag = int(round(cov * 100))
        out_dir = base / f"policy_cov{cov_tag}"
        m = run_train_eval_policy(features_csv, out_dir, cov, main_c_wrong, c_defer, group_col)
        coverage_rows.append(read_metrics(m, seed, {"target_coverage": cov, "cost_ratio": f"{main_c_wrong}:1", "c_wrong": main_c_wrong, "c_defer": c_defer}))
    coverage_out = pd.concat(coverage_rows, ignore_index=True)
    coverage_path = base / "coverage_comparison.csv"
    coverage_out.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    print("Saved:", coverage_path)

    # 2) matched coverage grid
    grid_rows = []
    for target in GRID_TARGETS:
        tag = int(round(target * 100))
        out_dir = base / f"policy_grid_{tag:02d}"
        m = run_train_eval_policy(features_csv, out_dir, target, main_c_wrong, c_defer, group_col)
        grid_rows.append(read_metrics(m, seed, {"target_coverage": target, "result_dir": path_s(out_dir), "cost_ratio": f"{main_c_wrong}:1", "c_wrong": main_c_wrong, "c_defer": c_defer}))
    grid = pd.concat(grid_rows, ignore_index=True)
    grid_path = base / "coverage_grid_all.csv"
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print("Saved:", grid_path)

    uaat = grid[grid["policy"] == "uaat_monotone"].copy()
    baselines = ["fixed", "uncertainty_grid"]
    matched_rows = []
    for _, u in uaat.iterrows():
        for baseline in baselines:
            cand = grid[grid["policy"] == baseline].copy()
            cand["coverage_gap"] = (cand["coverage"] - u["coverage"]).abs()
            b = cand.sort_values("coverage_gap").iloc[0]
            matched_rows.append({
                "seed": seed,
                "uaat_target": u["target_coverage"],
                "baseline": baseline,
                "baseline_target": b["target_coverage"],
                "cost_ratio": f"{main_c_wrong}:1",
                "c_wrong": main_c_wrong,
                "c_defer": c_defer,
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
    print("Saved:", matched_path)

    paper = matched[matched["uaat_target"].round(2).isin(PAPER_TARGETS)].copy()
    paper_path = base / "coverage_matched_paper_targets.csv"
    paper.to_csv(paper_path, index=False, encoding="utf-8-sig")
    print("Saved:", paper_path)

    # 3) cost sensitivity
    cost_rows = []
    for c_wrong in COSTS:
        for cov in COVERAGES:
            cov_tag = int(round(cov * 100))
            out_dir = base / f"cost_{c_wrong}to1_cov{cov_tag}"
            m = run_train_eval_policy(features_csv, out_dir, cov, c_wrong, c_defer, group_col)
            cost_rows.append(read_metrics(m, seed, {"cost_ratio": f"{c_wrong}:1", "c_wrong": c_wrong, "c_defer": c_defer, "target_coverage": cov}))
    cost_out = pd.concat(cost_rows, ignore_index=True)
    cost_path = base / "cost_sensitivity_comparison.csv"
    cost_out.to_csv(cost_path, index=False, encoding="utf-8-sig")
    print("Saved:", cost_path)
