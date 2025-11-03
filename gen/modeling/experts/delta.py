"""Utility wrappers for computing LoRA deltas."""

from __future__ import annotations

import torch
import torch.nn as nn


class PeftDeltaWrapper(nn.Module):
    """Wraps a PEFT model to output logits delta relative to a base model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, **model_kwargs) -> torch.Tensor:
        if "base_logits" not in model_kwargs:
            raise ValueError("base_logits must be provided")
        base_logits = model_kwargs.pop("base_logits")
        outputs = self.model(input_ids=input_ids, **model_kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits - base_logits
