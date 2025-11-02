"""Common interfaces for experts (top-level package)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, Tuple


class GatedExpertMixin(ABC):
    """Base interface for experts controlled by gating logic.

    语义约定：
    - forward_delta(x) 只返回增量 Δy（对 BASE 的修正量），不包含 y_base。
    - gate_inputs(...) 仅提取路由特征 z，不做打分。
    - gate_score(z) 根据 z 计算门控得分 s（如 Sigmoid/Softmax）。
    """

    @abstractmethod
    def forward_delta(self, x: Any, **kwargs: Any) -> Any:
        """Compute delta output Δy given inputs x (LoRA/Expert residual only)."""

    @abstractmethod
    def gate_inputs(self, *args: Any, **kwargs: Any) -> Any:
        """Extract routing features z (e.g., from hidden states or hardware embedding)."""

    @abstractmethod
    def gate_score(self, z: Any, **kwargs: Any) -> Any:
        """Compute gating score s from z (e.g., Sigmoid(W·z+b) or Softmax)."""

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize expert state for persistence."""

    @classmethod
    @abstractmethod
    def deserialize(cls, state: Dict[str, Any], *args: Any, **kwargs: Any) -> "GatedExpertMixin":
        """Restore an expert from serialized state."""


class ExpertRegistry:
    """Registry that keeps expert instances for gating."""

    def __init__(self):
        self._experts: "OrderedDict[str, GatedExpertMixin]" = OrderedDict()

    def register(self, name: str, expert: GatedExpertMixin) -> None:
        if not isinstance(expert, GatedExpertMixin):
            raise TypeError("expert must inherit from GatedExpertMixin")
        self._experts[name] = expert

    def get(self, name: str) -> GatedExpertMixin:
        if name not in self._experts:
            raise KeyError(f"Expert '{name}' is not registered")
        return self._experts[name]

    def unregister(self, name: str) -> None:
        if name in self._experts:
            self._experts.pop(name)

    def names(self) -> Iterable[str]:
        return list(self._experts.keys())

    def clear(self) -> None:
        self._experts.clear()

    def items(self) -> Iterable[Tuple[str, GatedExpertMixin]]:
        return list(self._experts.items())

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize all registered experts."""
        return {name: expert.serialize() for name, expert in self._experts.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Restore experts from serialized state. Caller必须已注册对应专家实例或占位符."""
        for name, expert_state in state.items():
            if name not in self._experts:
                raise KeyError(f"Expert '{name}' is not registered, cannot load state.")
            expert_cls = type(self._experts[name])
            self._experts[name] = expert_cls.deserialize(expert_state)
