"""EdgeTLM training loss helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_fn: Optional[torch.nn.Module] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """标准自回归交叉熵，兼容可选自定义 loss_fn。"""
    if loss_fn is not None:
        return loss_fn(logits, labels)

    if logits.dim() < 3:
        raise ValueError("logits must be [batch, seq, vocab] for causal LM loss.")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def compute_gain_loss(
    I_pct: torch.Tensor,
    g_mean: torch.Tensor,
    *,
    step: int,
    warmup_steps: int,
    m_target: float,
    lambda_gain: float,
    eps: float = 0.005,
) -> torch.Tensor:
    """门控收益损失：鼓励门控在收益低于阈值时收缩。"""
    if lambda_gain == 0.0:
        return torch.zeros((), device=I_pct.device, dtype=I_pct.dtype)

    ipct_mean = I_pct.mean()
    ipct_mean_value = float(ipct_mean.detach())

    apply_gain = (ipct_mean_value >= eps) or (step >= warmup_steps)
    if not apply_gain:
        return torch.zeros((), device=I_pct.device, dtype=I_pct.dtype)

    margin = 0.0 if step < warmup_steps else m_target
    penalty = F.relu(margin - ipct_mean)
    return lambda_gain * g_mean * penalty


def entropy_reg(gates: torch.Tensor, lambda_h: float, eps: float = 1e-6) -> torch.Tensor:
    """门控熵正则，鼓励稀疏激活。"""
    if lambda_h == 0.0:
        return torch.zeros((), device=gates.device, dtype=gates.dtype)

    gates = torch.clamp(gates, eps, 1.0 - eps)
    entropy = -(gates * torch.log(gates) + (1.0 - gates) * torch.log(1.0 - gates))
    return lambda_h * entropy.mean()


def l2r_reg(router: torch.Tensor, lambda_r: float) -> torch.Tensor:
    """路由向量 L2 正则。"""
    if lambda_r == 0.0:
        return torch.zeros((), device=router.device, dtype=router.dtype)
    return lambda_r * router.pow(2).mean()
