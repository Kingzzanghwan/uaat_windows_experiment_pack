from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PolicyTrainConfig:
    target_coverage: float = 0.80
    c_wrong: float = 5.0
    c_defer: float = 1.0
    steps: int = 1200
    lr: float = 0.02
    temp: float = 50.0
    coverage_weight: float = 20.0
    seed: int = 42


class MonotoneThresholdNet:
    """Light wrapper around a tiny PyTorch module.

    tau(x) = sigmoid(b + linear(context) + softplus(a_u) * uncertainty)
    softplus(a_u) >= 0, so tau is non-decreasing in uncertainty.
    """

    def __init__(self, context_cols: list[str], state_dict: dict[str, Any] | None = None):
        self.context_cols = context_cols
        self.state_dict = state_dict or {}

    def predict_tau(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        device = "cpu"
        ctx = self._context_matrix(df)
        u = df["uncertainty"].to_numpy(dtype=np.float32).reshape(-1, 1)
        sd = self.state_dict
        b = torch.tensor(sd["b"], dtype=torch.float32, device=device)
        w_u_raw = torch.tensor(sd["w_u_raw"], dtype=torch.float32, device=device)
        w_ctx = torch.tensor(sd["w_ctx"], dtype=torch.float32, device=device)
        x_ctx = torch.tensor(ctx, dtype=torch.float32, device=device)
        x_u = torch.tensor(u, dtype=torch.float32, device=device)
        logits = b + x_ctx.matmul(w_ctx.reshape(-1, 1)) + F.softplus(w_u_raw) * x_u
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    def _context_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if not self.context_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return df[self.context_cols].fillna(0.0).to_numpy(dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "monotone_policy_net",
            "context_cols": self.context_cols,
            "state_dict": self.state_dict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MonotoneThresholdNet":
        return cls(context_cols=list(d.get("context_cols", [])), state_dict=d["state_dict"])


def train_monotone_policy_net(
    calib: pd.DataFrame,
    context_cols: list[str],
    cfg: PolicyTrainConfig,
) -> MonotoneThresholdNet:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(cfg.seed)

    score = torch.tensor(calib["score"].to_numpy(dtype=np.float32).reshape(-1, 1))
    error = torch.tensor(calib["error"].to_numpy(dtype=np.float32).reshape(-1, 1))
    u = torch.tensor(calib["uncertainty"].to_numpy(dtype=np.float32).reshape(-1, 1))
    if context_cols:
        ctx_np = calib[context_cols].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        ctx_np = np.zeros((len(calib), 0), dtype=np.float32)
    ctx = torch.tensor(ctx_np)

    # NOTE: previously these were initialized to fixed constants (zeros / 0.5 / 0.0),
    # so torch.manual_seed(cfg.seed) above had nothing random to act on and every
    # seed converged to a bit-identical optimum. Small seed-controlled noise around
    # the same starting point restores genuine seed-to-seed variation for the
    # seed-expansion / statistical-significance experiment, while keeping the
    # original prior (b~0, w_ctx~0, w_u_raw~0.5) essentially unchanged.
    w_ctx = nn.Parameter(0.01 * torch.randn((ctx.shape[1], 1), dtype=torch.float32))
    w_u_raw = nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32) + 0.05 * torch.randn((1, 1), dtype=torch.float32))
    b = nn.Parameter(0.05 * torch.randn((1, 1), dtype=torch.float32))
    opt = torch.optim.Adam([w_ctx, w_u_raw, b], lr=cfg.lr)

    best_loss = float("inf")
    best_state = None
    has_ctx = ctx.shape[1] > 0  # empty when context is ablated (--use_inputs without ctx)
    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        ctx_term = ctx.matmul(w_ctx) if has_ctx else torch.zeros((ctx.shape[0], 1), dtype=torch.float32)
        tau = torch.sigmoid(b + ctx_term + F.softplus(w_u_raw) * u)
        p_accept = torch.sigmoid(cfg.temp * (score - tau))
        risk_loss = (p_accept * cfg.c_wrong * error + (1.0 - p_accept) * cfg.c_defer).mean()
        coverage_loss = (p_accept.mean() - cfg.target_coverage) ** 2
        # Guard the L2 penalty: mean() of an empty tensor is NaN, which would
        # poison the loss and prevent best_state from ever being set when
        # context is ablated. Use 0 in that case.
        ctx_penalty = (w_ctx ** 2).mean() if has_ctx else torch.zeros((), dtype=torch.float32)
        loss = risk_loss + cfg.coverage_weight * coverage_loss + 1e-4 * ctx_penalty
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "b": float(b.detach().cpu().numpy().reshape(-1)[0]),
                "w_u_raw": float(w_u_raw.detach().cpu().numpy().reshape(-1)[0]),
                "w_ctx": w_ctx.detach().cpu().numpy().reshape(-1).astype(float).tolist(),
                "best_loss": best_loss,
            }
    if best_state is None:
        # Fallback: if every step produced a non-finite loss (degenerate input),
        # still return the current parameters rather than crashing.
        best_state = {
            "b": float(b.detach().cpu().numpy().reshape(-1)[0]),
            "w_u_raw": float(w_u_raw.detach().cpu().numpy().reshape(-1)[0]),
            "w_ctx": w_ctx.detach().cpu().numpy().reshape(-1).astype(float).tolist(),
            "best_loss": float("inf"),
        }
    return MonotoneThresholdNet(context_cols=context_cols, state_dict=best_state)
