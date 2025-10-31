"""Utilities for freezing and wrapping the base TLM model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FrozenBaseWrapper(nn.Module):
    """Wrap a base model and freeze its parameters.

    - Freezes all parameters of the wrapped base model
    - Forwards all attributes to the underlying base model when not found
    - Keeps the base config accessible via the `config` property
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        if base_model is None:
            raise ValueError("base_model must not be None")
        self.base_model = base_model.eval()
        self.base_model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any):
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def config(self):
        return getattr(self.base_model, "config", None)

    @torch.no_grad()
    def router_feature(self, hidden_states=None, hw_emb=None):
        """Optional helper: pick routing feature from hidden states or hardware embedding."""
        if hw_emb is not None:
            return hw_emb
        assert hidden_states is not None
        return hidden_states.mean(dim=1)

