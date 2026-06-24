#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UAAT 논문용 '데이터 기반' 그림 자동 생성기 (E3 기준).
필요 패키지: numpy, pandas, matplotlib
  -> matplotlib 없으면:  pip install matplotlib

생성되는 그림 (figures\ 폴더):
  1) reliability_diagram.png : temperature scaling 전/후 신뢰도 보정 곡선
                               (각 패널에 ECE 표시 -> 보정 개선이 한눈에 보임)
  2) risk_coverage_e3.png    : 고정(fixed) 기준선의 위험-커버리지 곡선
                               + 운용범위 [0.60,0.90] 음영
  + 콘솔에 고정 기준선의 operating-range AURC 출력
    (논문 7.3절의 fixed 값 0.423112 와 비슷하게 나오면 계산 방식이 논문과 일치한다는 뜻)

사용법 (검은 창에서, venv 켜고):
  python uaat_make_figures.py --csv experiments\e3_tempscaled.csv ^
      --conf_col score --cal_col conf_tempscaled --error_col error --outdir figures
"""
import argparse, os
import numpy as np
# numpy 2.x 호환: np.trapz 가 사라지고 np.trapezoid 로 바뀜
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reliability(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    N = len(conf)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return e


def risk_at_coverage(score, error, cov, cw=5.0, cd=1.0):
    n = len(score)
    thr = np.quantile(score, 1 - cov)
    auto = score >= thr
    wrong_auto = np.logical_and(auto, error == 1).sum()
    defer = (~auto).sum()
    return cw * wrong_auto / n + cd * defer / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--conf_col", default="score")
    ap.add_argument("--cal_col", default="conf_tempscaled")
    ap.add_argument("--error_col", default="error")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--cost", default="5:1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    err = df[args.error_col].astype(float).values
    correct = 1.0 - err
    conf = df[args.conf_col].astype(float).values
    cw, cd = [float(x) for x in args.cost.split(":")]

    has_cal = args.cal_col in df.columns
    if not has_cal:
        print("[warn] '{}' 컬럼이 없어 temperature scaling '후' 패널은 raw로 대체합니다. "
              "temperature_scaling.py로 만든 e3_tempscaled.csv를 쓰세요.".format(args.cal_col))

    # ---------- Figure 1: reliability diagram (before / after) ----------
    panels = [(args.conf_col, "Before (raw score)")]
    panels.append((args.cal_col if has_cal else args.conf_col, "After temperature scaling"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (col, title) in zip(axes, panels):
        c = df[col].astype(float).values
        xs, ys = reliability(c, correct)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.bar(xs, ys, width=0.05, alpha=0.65, edgecolor="black", linewidth=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("{}\nECE = {:.3f}".format(title, ece(c, correct)))
    fig.suptitle("E3 (iWildCam) reliability diagram", y=1.02)
    fig.tight_layout()
    p1 = os.path.join(args.outdir, "reliability_diagram.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: risk-coverage curve (fixed baseline) ----------
    covs = np.linspace(0.50, 0.95, 46)
    risks = [risk_at_coverage(conf, err, c, cw, cd) for c in covs]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(covs, risks, "-o", ms=3, label="Fixed threshold")
    ax.axvspan(0.60, 0.90, color="orange", alpha=0.12, label="Operating range [0.60, 0.90]")
    ax.set_xlabel("Coverage (automation rate)")
    ax.set_ylabel("Decision risk (C = {}:{})".format(int(cw), int(cd)))
    ax.set_title("E3 risk-coverage curve (fixed baseline)")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(args.outdir, "risk_coverage_e3.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- operating-range AURC (fixed) for consistency check ----------
    g = np.linspace(0.60, 0.90, 31)
    aurc_fixed = float(np.trapz([risk_at_coverage(conf, err, c, cw, cd) for c in g], g))

    print("saved:", p1)
    print("saved:", p2)
    print("operating-range AURC (fixed, [0.60,0.90], trapezoid) = {:.6f}".format(aurc_fixed))
    print("  -> 논문 7.3절 fixed AURC(0.423112)와 비슷하면 계산 방식 일치 확인.")


if __name__ == "__main__":
    main()
