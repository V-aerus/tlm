"""Combination module for Frozen TLM base and gated experts."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_base import FrozenBaseWrapper
from .experts.interface import ExpertRegistry, GatedExpertMixin


def _broadcast_gate(gate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast gate weights to match target tensor rank."""
    while gate.ndim < target.ndim:
        gate = gate.unsqueeze(-1)
    return gate


class BasePlusExperts(nn.Module):
    """组合冻结 BASE 与若干 LoRA 专家，支持训练期/推理期前向。"""

    def __init__(self, base: FrozenBaseWrapper, registry: ExpertRegistry):
        super().__init__()
        if not isinstance(base, FrozenBaseWrapper):
            raise TypeError("base must be an instance of FrozenBaseWrapper")
        self.base = base
        self.registry = registry

    def forward_single(
        self,
        x: torch.Tensor,
        *,
        expert_name: str,
        hidden_states: Optional[torch.Tensor] = None,
        hw_emb: Optional[torch.Tensor] = None,
        base_kwargs: Optional[Dict] = None,
        expert_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """训练期：单专家前向，返回 y = y_base + σ(g)*Δy。"""
        if base_kwargs is None:
            base_kwargs = {}
        if expert_kwargs is None:
            expert_kwargs = {}

        expert = self.registry.get(expert_name)

        with torch.no_grad():
            y_base = self.base(x, **base_kwargs)

        z = expert.gate_inputs(hidden_states=hidden_states, hw_emb=hw_emb)
        g = expert.gating_weight(z)
        delta = expert.forward_delta(x, **expert_kwargs)

        return y_base + _broadcast_gate(g, delta) * delta

    @torch.no_grad()
    def forward_multi(
        self,
        x: torch.Tensor,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        hw_emb: Optional[torch.Tensor] = None,
        topk: int = 2,
        tau_override: Optional[float] = None,
        mask: Optional[Dict[str, bool]] = None,
        base_kwargs: Optional[Dict] = None,
        expert_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """推理期：按 Top-K 稀疏路由融合多个专家。"""
        if base_kwargs is None:
            base_kwargs = {}
        if expert_kwargs is None:
            expert_kwargs = {}

        y_base = self.base(x, **base_kwargs)

        scores = []
        deltas = []
        names = []

        for name, expert in self.registry.items():
            if mask and not mask.get(name, True):
                continue
            z = expert.gate_inputs(hidden_states=hidden_states, hw_emb=hw_emb)
            s = expert.gate_score(z)
            tau = tau_override if tau_override is not None else float(expert.temperature())
            scores.append(s / tau)
            deltas.append(expert.forward_delta(x, **expert_kwargs))
            names.append(name)

        if not scores:
            return y_base, {"weights": torch.ones((1, y_base.size(0)), device=y_base.device), "names": ["BASE"]}

        stacked_scores = torch.stack(scores, dim=0)  # [E, B]
        base_row = torch.zeros_like(stacked_scores[0:1])  # [1, B]
        full_scores = torch.cat([base_row, stacked_scores], dim=0)  # [1+E, B]

        k = min(topk, stacked_scores.size(0))
        topk_vals, topk_idx = torch.topk(stacked_scores, k=k, dim=0)

        mask_tensor = torch.full_like(full_scores, fill_value=-1e9)
        mask_tensor[0] = 0.0  # BASE 通道保持 0 分

        batch_indices = torch.arange(full_scores.size(1), device=full_scores.device)
        for rank in range(topk_idx.size(0)):
            expert_indices = topk_idx[rank]
            mask_tensor[expert_indices + 1, batch_indices] = topk_vals[rank, batch_indices]

        weights = F.softmax(mask_tensor, dim=0)

        y = y_base
        for idx, delta in enumerate(deltas):
            gate = _broadcast_gate(weights[idx + 1], delta)
            y = y + gate * delta

        return y, {"weights": weights, "names": ["BASE"] + names}
