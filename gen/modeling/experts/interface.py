"""Common interfaces for experts (top-level package)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, TypeVar


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


T = TypeVar("T", bound=GatedExpertMixin)


class ExpertRegistry:
    """Simple registry for mapping expert names to their implementations."""

    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str, expert_cls: Type[T]) -> None:
        if not issubclass(expert_cls, GatedExpertMixin):
            raise TypeError("expert_cls must inherit from GatedExpertMixin")
        self._registry[name] = expert_cls

    def get(self, name: str) -> Type[T]:
        if name not in self._registry:
            raise KeyError(f"Expert '{name}' is not registered")
        return self._registry[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        expert_cls = self.get(name)
        return expert_cls(*args, **kwargs)

    def registered(self) -> List[str]:
        return sorted(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()


