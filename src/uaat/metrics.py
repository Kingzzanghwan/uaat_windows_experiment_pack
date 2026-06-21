from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DecisionMetrics:
    coverage: float
    risk: float
    wrong_auto_rate: float
    auto_error_rate: float
    defer_rate: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "coverage": self.coverage,
            "risk": self.risk,
            "wrong_auto_rate": self.wrong_auto_rate,
            "auto_error_rate": self.auto_error_rate,
            "defer_rate": self.defer_rate,
            "n": self.n,
        }


def decision_metrics(
    df: pd.DataFrame,
    accept: np.ndarray,
    c_wrong: float = 5.0,
    c_defer: float = 1.0,
) -> DecisionMetrics:
    accept = np.asarray(accept).astype(bool)
    error = df["error"].to_numpy().astype(bool)
    n = len(df)
    if n == 0:
        raise ValueError("Empty dataframe passed to decision_metrics().")
    coverage = float(accept.mean())
    defer_rate = float(1.0 - coverage)
    wrong_auto_rate = float((accept & error).mean())
    auto_error_rate = float((accept & error).sum() / max(1, accept.sum()))
    risk = float(c_wrong * wrong_auto_rate + c_defer * defer_rate)
    return DecisionMetrics(
        coverage=coverage,
        risk=risk,
        wrong_auto_rate=wrong_auto_rate,
        auto_error_rate=auto_error_rate,
        defer_rate=defer_rate,
        n=n,
    )


def threshold_for_coverage(scores: np.ndarray, target_coverage: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if not (0.0 < target_coverage < 1.0):
        raise ValueError("target_coverage must be between 0 and 1, e.g. 0.80")
    return float(np.quantile(scores, 1.0 - target_coverage))


def fixed_threshold_policy(calib: pd.DataFrame, target_coverage: float) -> dict[str, Any]:
    tau = threshold_for_coverage(calib["score"].to_numpy(), target_coverage)
    return {"name": "fixed", "threshold": tau}


def apply_fixed(df: pd.DataFrame, policy: dict[str, Any]) -> np.ndarray:
    return df["score"].to_numpy() > float(policy["threshold"])


def uncertainty_grid_policy(
    calib: pd.DataFrame,
    target_coverage: float,
    c_wrong: float = 5.0,
    c_defer: float = 1.0,
    max_alpha: float = 3.0,
    steps: int = 121,
) -> dict[str, Any]:
    """Find score - alpha * uncertainty threshold.

    Decision: score - alpha*u > theta, equivalent to score > theta + alpha*u.
    Therefore the effective threshold is monotone non-decreasing in u when alpha >= 0.
    """
    best: dict[str, Any] | None = None
    score = calib["score"].to_numpy(dtype=np.float64)
    u = calib["uncertainty"].to_numpy(dtype=np.float64)
    for alpha in np.linspace(0.0, max_alpha, steps):
        adjusted = score - alpha * u
        theta = threshold_for_coverage(adjusted, target_coverage)
        accept = adjusted > theta
        m = decision_metrics(calib, accept, c_wrong=c_wrong, c_defer=c_defer)
        item = {
            "name": "uncertainty_grid",
            "alpha": float(alpha),
            "theta": float(theta),
            "calib_risk": m.risk,
            "calib_coverage": m.coverage,
        }
        if best is None or item["calib_risk"] < best["calib_risk"]:
            best = item
    assert best is not None
    return best


def apply_uncertainty_grid(df: pd.DataFrame, policy: dict[str, Any]) -> np.ndarray:
    adjusted = df["score"].to_numpy(dtype=np.float64) - float(policy["alpha"]) * df[
        "uncertainty"
    ].to_numpy(dtype=np.float64)
    return adjusted > float(policy["theta"])
