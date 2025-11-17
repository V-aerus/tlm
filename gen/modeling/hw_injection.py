"""Hardware embedding injection modules (e.g., ProtoMix aligner)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoMixAligner(nn.Module):
    """Project hardware vectors into the model embedding space via convex mixing."""

    def __init__(
        self,
        prototype_keys: torch.Tensor,
        embed_dim: int,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(prototype_keys, torch.Tensor):
            prototype_keys = torch.tensor(prototype_keys, dtype=torch.float32)
        if prototype_keys.ndim != 2:
            raise ValueError("prototype_keys must be 2-D (num_proto, key_dim)")

        proto = F.normalize(prototype_keys.float(), dim=-1)
        self.register_buffer("prototype_keys", proto)
        num_proto = proto.size(0)
        self.embed_dim = embed_dim

        self.prototypes = nn.Parameter(torch.randn(num_proto, embed_dim) * 0.02)

        if trainable_temperature:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))))
            self.register_buffer("temperature", torch.tensor(float(temperature)))
        else:
            self.log_temp = None
            self.register_buffer("temperature", torch.tensor(float(temperature)))

    def get_temperature(self) -> torch.Tensor:
        if self.log_temp is not None:
            return torch.exp(self.log_temp)
        return self.temperature

    def forward(self, hw_vec: torch.Tensor, return_weights: bool = False):
        """Map hardware vectors (B, D_key) to embedding vectors (B, embed_dim)."""

        if hw_vec.ndim != 2:
            raise ValueError("hw_vec must be 2-D (batch, key_dim)")
        proto = self.prototype_keys
        hw_norm = F.normalize(hw_vec, dim=-1)
        sim = torch.matmul(hw_norm, proto.t())  # (B, num_proto)
        temp = self.get_temperature()
        weights = torch.softmax(sim / temp, dim=-1)
        mix = torch.matmul(weights, self.prototypes)
        if return_weights:
            return mix, weights
        return mix

    def extra_repr(self) -> str:
        return (
            f"num_proto={self.prototype_keys.size(0)}, "
            f"embed_dim={self.embed_dim}, "
            f"trainable_temperature={self.log_temp is not None}"
        )


def build_prototype_matrix(embeddings: dict, prototype_names: list[str]) -> torch.Tensor:
    """Utility to assemble prototype matrix from embedding dict."""

    vectors = []
    for name in prototype_names:
        if name not in embeddings:
            raise KeyError(f"Prototype '{name}' not found in embedding dict")
        vectors.append(torch.tensor(embeddings[name], dtype=torch.float32))
    return torch.stack(vectors, dim=0)
