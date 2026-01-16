"""Gated LoRA expert implementation for EdgeTLM."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import GatedExpertMixin


def _softplus_inverse(value: float) -> float:
    """Numerically stable inverse of softplus for scalar values."""
    value_tensor = torch.tensor(value, dtype=torch.float32)
    eps = torch.finfo(value_tensor.dtype).eps
    return torch.log(torch.expm1(torch.clamp(value_tensor, min=eps)))


class GatedLoRAExpert(nn.Module, GatedExpertMixin):
    """EdgeTLM 单行路由 LoRA 专家。

    参数:
        lora_module: 负责输出 Δy 的 LoRA 模块（已按目标层包装好）。
        r_dim: 路由向量维度。
        tau_init: 温度初值。
        b_init: 负偏置初值（<=0）。
    """

    def __init__(
        self,
        lora_module: nn.Module,
        r_dim: int,
        tau_init: float = 2.0,
        b_init: float = -1.0,
        init_router: Optional[torch.Tensor] = None,
        reset_lora: bool = True,
    ):
        super().__init__()
        if lora_module is None:
            raise ValueError("lora_module must not be None")
        if r_dim <= 0:
            raise ValueError("r_dim must be positive")

        self.lora = lora_module

        self.r = nn.Parameter(torch.zeros(r_dim, dtype=torch.float32))
        if init_router is not None:
            if init_router.shape[-1] != r_dim:
                raise ValueError("init_router dimensionality mismatch.")
            with torch.no_grad():
                self.r.copy_(init_router)

        # beta 参数化负偏置：b = -softplus(beta) <= 0
        target = max(float(-b_init), 1e-6)
        beta_init = _softplus_inverse(target)
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

        self.register_buffer("tau", torch.tensor(float(tau_init), dtype=torch.float32))

        if reset_lora:
            self._reset_lora_parameters()

    def _reset_lora_parameters(self) -> None:
        """确保 LoRA B 矩阵初始为 0，保持首轮前向稳定。"""
        for name, param in self.lora.named_parameters():
            if "lora_B" in name:
                nn.init.zeros_(param)

    def forward_delta(self, x: Any, **kwargs: Any) -> Any:
        return self.lora(x, **kwargs)

    def gate_inputs(self, *, hidden_states: Optional[torch.Tensor] = None, hw_emb: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        if hw_emb is not None:
            return hw_emb
        if hidden_states is None:
            raise ValueError("Either hidden_states or hw_emb must be provided.")
        return hidden_states.mean(dim=1)

    def gate_score(self, z: torch.Tensor, **_: Any) -> torch.Tensor:
        b = -F.softplus(self.beta)
        return (z * self.r).sum(dim=-1) + b

    def temperature(self) -> torch.Tensor:
        return self.tau

    def gating_weight(self, z: torch.Tensor) -> torch.Tensor:
        s = self.gate_score(z)
        return torch.sigmoid(s / self.tau)

    def serialize(self) -> Dict[str, Any]:
        return {
            "lora": self.lora.state_dict(),
            "r": self.r.detach().cpu().tolist(),
            "beta": float(self.beta.detach().cpu()),
            "tau": float(self.tau.detach().cpu()),
        }

    @classmethod
    def deserialize(cls, state: Dict[str, Any], *, lora_module: nn.Module) -> "GatedLoRAExpert":
        if lora_module is None:
            raise ValueError("lora_module is required to deserialize GatedLoRAExpert.")
        r_tensor = torch.tensor(state["r"], dtype=torch.float32)
        tau = state.get("tau", 2.0)
        beta = state.get("beta", -1.0)
        expert = cls(
            lora_module=lora_module,
            r_dim=r_tensor.numel(),
            tau_init=float(tau),
            b_init=-1.0,
        )
        expert.lora.load_state_dict(state["lora"])
        with torch.no_grad():
            expert.r.copy_(r_tensor)
            expert.beta.fill_(float(beta))
            expert.tau.fill_(float(tau))
        return expert
