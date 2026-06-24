#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temperature scaling baseline for UAAT calibration.
Dependencies: numpy, pandas only (no scipy needed).

------------------------------------------------------------------
사용법 (2단계)
------------------------------------------------------------------
1) INSPECT (컬럼 확인) -- 옵션 없이 CSV만:
     python experiments\\temperature_scaling.py --csv "PATH_TO.csv"
   -> 컬럼/타입/통계/이진(0,1)후보 컬럼을 출력. 여기서 conf_col, error_col을 고른다.

2) RUN (실제 학습+보정):
     python experiments\\temperature_scaling.py --csv "TEST.csv" ^
         --conf_col <confidence컬럼> --error_col <error컬럼> ^
         --fit_csv "VAL_OR_TRAIN.csv" ^
         --out_csv "experiments\\e3_tempscaled.csv"
   -> 학습된 T, ECE/NLL before/after를 출력하고,
      conf_tempscaled 컬럼이 추가된 새 CSV를 저장한다.

학습/평가 분리:
  - --fit_csv 를 주면 그 데이터로 T를 학습하고, --csv(test)에서 평가한다 (정석).
  - --fit_csv 를 안 주면 --csv 한 곳에서 학습+평가 (약한 낙관 편향, 경고 출력).

error_col 해석:
  - {0,1} 또는 bool (1 = 틀림) -> 정답 y = 1 - error  (자동)
  - 연속값(loss 등)이면 --error_threshold THR 로 이진화 (error > THR 이면 틀림)
"""
import argparse
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def nll(p, y, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def ece_equal_width(conf, correct, n_bins=15):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=False), 0, n_bins - 1)
    N = len(conf)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        ece += (m.sum() / N) * abs(acc - avg_conf)
    return float(ece)


def fit_temperature(z, y):
    """1D minimization of NLL over T>0 for p = sigmoid(z / T). Grid + refine, no scipy."""
    grid = np.concatenate([np.linspace(0.05, 1.0, 40), np.linspace(1.0, 12.0, 110)])
    best_T, best_v = 1.0, np.inf
    for T in grid:
        v = nll(sigmoid(z / T), y)
        if v < best_v:
            best_v, best_T = v, T
    lo, hi = best_T * 0.6, best_T * 1.5
    for T in np.linspace(lo, hi, 300):
        v = nll(sigmoid(z / T), y)
        if v < best_v:
            best_v, best_T = v, T
    return float(best_T)


def inspect(df, csv_path):
    print("=== INSPECT MODE (no --conf_col/--error_col given) ===")
    print("File:", csv_path, "| rows:", len(df))
    print("\n[Columns]")
    for c in df.columns:
        print("  {:30s} dtype={}".format(str(c), df[c].dtype))
    print("\n[Head]")
    print(df.head().to_string())
    num = df.select_dtypes(include=[np.number])
    if len(num.columns) > 0:
        print("\n[Numeric summary]  (min/max로 confidence가 [0,1]인지 확인)")
        print(num.describe().T[["min", "max", "mean"]].to_string())
    print("\n[<=4 unique 값인 컬럼 = error/correct 후보]")
    for c in df.columns:
        u = pd.unique(df[c].dropna())
        if len(u) <= 4:
            try:
                us = sorted(u.tolist())
            except Exception:
                us = list(u)[:6]
            print("  {}: {}".format(c, us[:6]))
    print("\n다음 단계: 위에서 confidence 컬럼과 error/correct 컬럼을 골라")
    print("  --conf_col <conf> --error_col <error>  를 붙여 다시 실행하세요.")


def prep(df, conf_col, error_col, error_threshold):
    conf = df[conf_col].astype(float).values
    err_raw = df[error_col].values
    errf = pd.to_numeric(pd.Series(err_raw), errors="coerce").values
    if error_threshold is not None:
        wrong = (errf > error_threshold).astype(float)
        y = 1.0 - wrong
    else:
        uniq = set(np.unique(errf[~np.isnan(errf)]).tolist())
        if not uniq.issubset({0.0, 1.0}):
            raise SystemExit(
                "error_col '{}' 가 {{0,1}} 이진이 아닙니다 (예: {}). "
                "--error_threshold THR 로 이진화하세요.".format(error_col, list(uniq)[:5])
            )
        y = 1.0 - errf
    return conf, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="평가(test) CSV")
    ap.add_argument("--fit_csv", default=None, help="T 학습용 CSV (없으면 --csv로 학습+평가)")
    ap.add_argument("--conf_col", default=None)
    ap.add_argument("--error_col", default=None)
    ap.add_argument("--error_threshold", type=float, default=None,
                    help="연속 error 이진화: error > THR 이면 틀림")
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--out_csv", default="e3_tempscaled.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.conf_col is None or args.error_col is None:
        inspect(df, args.csv)
        return

    fit_df = pd.read_csv(args.fit_csv) if args.fit_csv else df
    if args.fit_csv is None:
        print("[warn] --fit_csv 미지정: 같은 셋에서 학습+평가합니다 (약한 낙관 편향).")
        print("       논문용으로는 val/train에서 학습하고 test에서 평가하는 게 정석입니다.\n")

    conf_fit, y_fit = prep(fit_df, args.conf_col, args.error_col, args.error_threshold)
    conf_eval, y_eval = prep(df, args.conf_col, args.error_col, args.error_threshold)

    cmin, cmax = np.nanmin(conf_fit), np.nanmax(conf_fit)
    in01 = (cmin >= 0.0) and (cmax <= 1.0)
    if in01:
        z_fit = logit(conf_fit)
        z_eval = logit(conf_eval)
        base_eval_prob = np.clip(conf_eval, 1e-6, 1 - 1e-6)
        mode = "confidence in [0,1] -> 정석 temperature scaling: p_T = sigmoid(logit(p)/T)"
    else:
        mu = np.nanmean(conf_fit)
        sd = np.nanstd(conf_fit) + 1e-12
        z_fit = (conf_fit - mu) / sd
        z_eval = (conf_eval - mu) / sd
        base_eval_prob = sigmoid(z_eval)
        mode = ("raw score (not in [0,1]) -> 표준화 후 단일파라미터 로지스틱(Platt-style) "
                "temperature. baseline ECE는 표준화 proxy 기준임에 주의")

    T = fit_temperature(z_fit, y_fit)
    p_cal_eval = sigmoid(z_eval / T)

    ece_before = ece_equal_width(base_eval_prob, y_eval, args.n_bins)
    ece_after = ece_equal_width(p_cal_eval, y_eval, args.n_bins)
    nll_before = nll(base_eval_prob, y_eval)
    nll_after = nll(p_cal_eval, y_eval)

    out = df.copy()
    out["conf_tempscaled"] = p_cal_eval
    out.to_csv(args.out_csv, index=False)

    print("==== Temperature Scaling 결과 ====")
    print("mode          :", mode)
    print("fit  data     : {} (n={})".format(args.fit_csv or args.csv, len(fit_df)))
    print("eval data     : {} (n={})".format(args.csv, len(df)))
    print("learned T     : {:.4f}".format(T))
    print("n_bins        : {}".format(args.n_bins))
    print("ECE  before   : {:.6f}".format(ece_before))
    print("ECE  after(T) : {:.6f}".format(ece_after))
    print("NLL  before   : {:.6f}".format(nll_before))
    print("NLL  after(T) : {:.6f}".format(nll_after))
    print("saved         : {}  (새 컬럼: conf_tempscaled)".format(args.out_csv))
    print()
    direction = "낮춤" if ece_after < ece_before else ("거의 동일" if abs(ece_after - ece_before) < 5e-3 else "오히려 높임")
    print("논문 한 줄 초안 (문구는 다듬으세요):")
    print("  Post-hoc temperature scaling (T={:.2f}) changes ECE from {:.3f} to {:.3f} "
          "on the E3 test split ({}).".format(T, ece_before, ece_after, direction))
    print()
    print(">> 다음 단계: 논문과 '같은 ECE 측정 방식'으로 비교하려면 아래를 실행:")
    print('   python experiments\\analysis_calibration.py --csv "{}" '
          '--conf_col conf_tempscaled --error_col {} --out_csv experiments\\ece_after.csv '
          '--out_png experiments\\ece_after.png'.format(args.out_csv, args.error_col))


if __name__ == "__main__":
    main()
