"""Combination module for Frozen TLM base and gated experts."""

from __future__ import annotations

import os
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
        self._debug_topk_once = False

    def forward_single(
        self,
        x: torch.Tensor,
        *,
        expert_name: str,
        hidden_states: Optional[torch.Tensor] = None,
        hw_emb: Optional[torch.Tensor] = None,
        base_kwargs: Optional[Dict] = None,
        expert_kwargs: Optional[Dict] = None,
        cached_base: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """训练期：单专家前向，返回 y = y_base + σ(g)*Δy。"""
        if base_kwargs is None:
            base_kwargs = {}
        if expert_kwargs is None:
            expert_kwargs = {}

        expert = self.registry.get(expert_name)

        if cached_base is None:
            with torch.no_grad():
                y_base = self.base(x, **base_kwargs)
        else:
            y_base = cached_base

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
        cached_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """推理期：按 Top-K 稀疏路由融合多个专家。"""
        if base_kwargs is None:
            base_kwargs = {}
        if expert_kwargs is None:
            expert_kwargs = {}

        if cached_base is None:
            y_base = self.base(x, **base_kwargs)
        else:
            y_base = cached_base

        scores = []
        experts = []
        names = []
        g_scores = []
        use_two_stage = False
        comp_mode = None
        comp_tau = None

        for name, expert in self.registry.items():
            if mask and not mask.get(name, True):
                continue
            z = expert.gate_inputs(hidden_states=hidden_states, hw_emb=hw_emb)
            s = expert.gate_score(z)
            tau = tau_override if tau_override is not None else float(expert.temperature())
            scores.append(s / tau)
            g_scores.append(torch.sigmoid(s / tau))
            if getattr(expert, "router_two_stage", False):
                use_two_stage = True
            mode = getattr(expert, "router_score_mode", None)
            if mode:
                comp_mode = mode
            ct = getattr(expert, "router_competition_tau", None)
            if ct is not None:
                comp_tau = float(ct)
            experts.append(expert)
            names.append(name)

        if not scores:
            return y_base, {"weights": torch.ones((1, y_base.size(0)), device=y_base.device), "names": ["BASE"]}

        stacked_scores = torch.stack(scores, dim=0)  # [E, B]
        if not use_two_stage and not comp_mode:
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
        else:
            # Two-stage routing: strength gate (sigmoid) + competition (similarity).
            comp_mode = comp_mode or "dot_over_rnorm"
            comp_tau = comp_tau if comp_tau is not None else 1.0
            g_stack = torch.stack(g_scores, dim=0)  # [E, B]

            comp_scores = []
            for expert, score in zip(experts, scores):
                z = expert.gate_inputs(hidden_states=hidden_states, hw_emb=hw_emb)
                r_dir = getattr(expert, "router_direction", None)
                if callable(r_dir):
                    r_vec = r_dir()
                else:
                    r_vec = getattr(expert, "r", None)
                if r_vec is None:
                    comp_scores.append(score)
                    continue
                dot = (z * r_vec).sum(dim=-1)
                r_norm = torch.linalg.norm(r_vec)
                if comp_mode == "cosine":
                    h_norm = torch.linalg.norm(z, dim=-1)
                    comp = dot / ((r_norm + 1e-6) * (h_norm + 1e-6))
                elif comp_mode == "dot":
                    comp = dot
                else:
                    comp = dot / (r_norm + 1e-6)
                comp_scores.append(comp)

            comp_stack = torch.stack(comp_scores, dim=0)  # [E, B]
            k = min(topk, comp_stack.size(0))
            topk_vals, topk_idx = torch.topk(comp_stack, k=k, dim=0)
            mask_tensor = torch.full_like(comp_stack, fill_value=-1e9)
            batch_indices = torch.arange(comp_stack.size(1), device=comp_stack.device)
            for rank in range(topk_idx.size(0)):
                expert_indices = topk_idx[rank]
                mask_tensor[expert_indices, batch_indices] = topk_vals[rank, batch_indices]
            pi = F.softmax(mask_tensor / max(comp_tau, 1e-6), dim=0)
            w = g_stack * pi
            base_weight = 1.0 - w.sum(dim=0)
            base_weight = torch.clamp(base_weight, min=0.0, max=1.0)
            weights = torch.cat([base_weight.unsqueeze(0), w], dim=0)

        y = y_base
        if k > 0:
            selected_indices = torch.unique(topk_idx).tolist()
            if os.environ.get("EDGE_EXPERT_DEBUG_TOPK") == "1" and not self._debug_topk_once:
                selected_names = [names[idx] for idx in selected_indices]
                print(f"[BasePlusExperts] topk_selected_indices={selected_indices} names={selected_names}")
                self._debug_topk_once = True
            for idx in selected_indices:
                delta = experts[idx].forward_delta(x, **expert_kwargs)
                gate = _broadcast_gate(weights[idx + 1], delta)
                y = y + gate * delta

        return y, {"weights": weights, "names": ["BASE"] + names}
