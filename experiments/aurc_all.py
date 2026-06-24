#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
운용범위 AURC (operating-range AURC, [0.60, 0.90]) 계산기.
필요 패키지: numpy, pandas  (matplotlib 불필요)

이 스크립트가 계산하는 것:
  - fixed (고정 임계값):           점수(score)로 순위/임계
  - uncertainty-grid (불확실성-격자): margin = score - alpha*uncertainty 로 임계
                                    (alpha를 격자 탐색하여 AURC 최소가 되는 값 보고)

주의(중요):
  - UAAT 자체의 AURC는 '표본별 임계값 tau(x)'가 필요하므로 features.csv만으로는
    계산할 수 없습니다(여기엔 score/uncertainty/error만 있음). UAAT의 AURC는
    논문 7.3절의 E3 값(0.422742)을 만든 '기존 평가 파이프라인'을 E1/E2에 그대로
    돌려서 얻으세요. (가이드 문서 ''필수 1'' 참조)
  - 따라서 이 스크립트는 (a) fixed/uncertainty-grid 두 값을 빠르게 확인하고,
    (b) 계산 방식이 논문과 일치하는지(E3 fixed가 0.4231 근처로 나오는지) 교차검증
    하는 용도입니다.

사용법 (검은 창, venv 켜고):
  python aurc_all.py --csv runs\\e1_cifar10c\\features.csv ^
      --conf_col score --unc_col uncertainty --error_col error --split_col split --test_value test
  python aurc_all.py --csv runs\\e2_mvtec_ad\\features.csv ^
      --conf_col score --unc_col uncertainty --error_col error --split_col split --test_value test
  (E3로 교차검증:) python aurc_all.py --csv runs\\e3_iwildcam\\features.csv ^
      --conf_col score --unc_col uncertainty --error_col error --split_col split --test_value test

컬럼 이름을 모르면 먼저:
  python temperature_scaling.py --csv <그 features.csv>   (INSPECT 모드로 컬럼 확인)
"""
import argparse
import numpy as np
# numpy 2.x 호환: np.trapz 가 사라지고 np.trapezoid 로 바뀜
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import pandas as pd


def risk_at_coverage(margin, error, cov, cw=5.0, cd=1.0):
    """margin 기준 (1-cov) 분위수로 임계, 커버리지-매칭 위험 계산."""
    n = len(margin)
    thr = np.quantile(margin, 1.0 - cov)
    auto = margin >= thr
    wrong_auto = np.logical_and(auto, error == 1).sum()
    defer = (~auto).sum()
    return cw * wrong_auto / n + cd * defer / n


def operating_range_aurc(margin, error, lo=0.60, hi=0.90, step=0.01, cw=5.0, cd=1.0):
    """[lo,hi] 구간에서 위험-커버리지 곡선을 사다리꼴 적분."""
    covs = np.round(np.arange(lo, hi + 1e-9, step), 4)
    risks = np.array([risk_at_coverage(margin, error, c, cw, cd) for c in covs])
    # 사다리꼴 적분 후 구간 폭으로 정규화하지 않은 '면적' (논문과 동일 관례)
    area = np.trapz(risks, covs)
    return area, covs, risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--conf_col", default="score")
    ap.add_argument("--unc_col", default="uncertainty")
    ap.add_argument("--error_col", default="error")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--test_value", default="test",
                    help="테스트 행을 고르는 split 값(없으면 --split_col '' 로 두면 전체 사용)")
    ap.add_argument("--lo", type=float, default=0.60)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cw", type=float, default=5.0)
    ap.add_argument("--cd", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.split_col and args.split_col in df.columns and args.test_value:
        df = df[df[args.split_col].astype(str) == args.test_value].copy()

    score = df[args.conf_col].astype(float).values
    unc = df[args.unc_col].astype(float).values
    error = pd.to_numeric(df[args.error_col], errors="coerce").fillna(0).astype(int).values
    n = len(df)

    # fixed: margin = score
    aurc_fixed, _, _ = operating_range_aurc(score, error, args.lo, args.hi, args.step, args.cw, args.cd)

    # uncertainty-grid: margin = score - alpha*unc, alpha 격자 탐색
    su = unc.std() + 1e-12
    alphas = np.concatenate([[0.0], np.linspace(0.05, 3.0, 60)]) * (score.std() / su)
    best_alpha, best_aurc = 0.0, np.inf
    for a in alphas:
        margin = score - a * unc
        au, _, _ = operating_range_aurc(margin, error, args.lo, args.hi, args.step, args.cw, args.cd)
        if au < best_aurc:
            best_aurc, best_alpha = au, a

    print("==== operating-range AURC [{:.2f},{:.2f}]  (낮을수록 좋음) ====".format(args.lo, args.hi))
    print("file            :", args.csv)
    print("test rows (n)   :", n)
    print("cost ratio      : {:.0f}:{:.0f}".format(args.cw, args.cd))
    print("-")
    print("fixed            AURC = {:.6f}".format(aurc_fixed))
    print("uncertainty-grid AURC = {:.6f}   (best alpha = {:.4f})".format(best_aurc, best_alpha))
    print("-")
    print(">> UAAT의 AURC는 features.csv만으로 계산 불가(표본별 tau 필요).")
    print("   논문 7.3절 E3 값(UAAT=0.422742)을 만든 기존 평가 스크립트를 E1/E2에 동일 실행해 채우세요.")
    print()
    print("[교차검증 팁] 이 파일이 E3라면 위 fixed 값이 논문의 0.423112 근처여야 계산 방식이 일치합니다.")
    print("              (분위수/적분 step 차이로 소수점 뒤가 약간 다를 수 있음)")


if __name__ == "__main__":
    main()
