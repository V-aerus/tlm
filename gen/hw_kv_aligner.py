import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HwKVAligner(nn.Module):
    """
    Hardware KV Aligner (mPnP-style KV injection).
    Maps a hardware embedding to a small number of KV slots (default 4),
    and injects them into the last `backward_depth` layers via per-layer linker weights.
    """

    def __init__(
        self,
        llm_num_layers: int,
        llm_num_heads: int,
        llm_head_dim: int,
        hw_dim: int = 24,
        num_slots: int = 4,
        backward_depth: int = 4,
        linker_temperature: float = 1.0,
        kv_scale_init: float = 0.01,
    ):
        super().__init__()
        self.llm_num_layers = llm_num_layers
        self.llm_num_heads = llm_num_heads
        self.llm_head_dim = llm_head_dim
        self.hw_dim = hw_dim
        self.num_slots = num_slots
        self.backward_depth = min(backward_depth, llm_num_layers)
        self.linker_temperature = linker_temperature

        d_model = llm_num_heads * llm_head_dim
        self.hw_mlp = nn.Sequential(
            nn.Linear(hw_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_slots * d_model),
        )
        self.key_aligner = nn.Linear(llm_head_dim, llm_head_dim)
        self.value_aligner = nn.Linear(llm_head_dim, llm_head_dim)
        self.linker_weights = nn.Parameter(torch.zeros(backward_depth))
        self.kv_scale = nn.Parameter(torch.tensor(float(kv_scale_init)))

    def forward(self, hw_emb: torch.Tensor, batch_size: int, num_beams: int = 1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        hw_emb: [B, hw_dim]
        batch_size: text batch size
        num_beams: beam size for generation

        return: tuple of past_key_values length = llm_num_layers;
                each element: (key, value) where shape = [B*num_beams, H, T_hw, d_head]
        """
        if hw_emb.dim() != 2 or hw_emb.size(-1) != self.hw_dim:
            raise ValueError(f"hw_emb shape {hw_emb.shape} is invalid; expect [B, {self.hw_dim}]")
        B = batch_size
        # Project hardware embedding to KV tokens
        x = self.hw_mlp(hw_emb)  # [B, num_slots*H*d]
        x = x.view(B, self.num_slots, self.llm_num_heads, self.llm_head_dim)  # [B, T, H, d]
        x = x.permute(0, 2, 1, 3)  # [B, H, T, d]
        mm_key = self.key_aligner(x)
        mm_value = self.value_aligner(x)
        if num_beams > 1:
            mm_key = mm_key.repeat_interleave(num_beams, dim=0)  # [B*num_beams, H, T, d]
            mm_value = mm_value.repeat_interleave(num_beams, dim=0)

        # linker weights for last backward_depth layers
        weights = F.softmax(self.linker_weights / max(self.linker_temperature, 1e-6), dim=0)
        past = []
        zero_key = torch.zeros_like(mm_key)
        zero_value = torch.zeros_like(mm_value)
        start_inject = max(0, self.llm_num_layers - self.backward_depth)
        for layer_idx in range(self.llm_num_layers):
            if layer_idx < start_inject:
                past.append((zero_key, zero_value))
            else:
                w = weights[layer_idx - start_inject]
                k = mm_key * w * self.kv_scale
                v = mm_value * w * self.kv_scale
                past.append((k, v))
        return tuple(past)


if __name__ == "__main__":
    # Self-test
    aligner = HwKVAligner(llm_num_layers=24, llm_num_heads=16, llm_head_dim=64, hw_dim=24, num_slots=4)
    hw = torch.randn(1, 24)
    out = aligner(hw, batch_size=2, num_beams=4)
    print("type:", type(out), "layers:", len(out))
    print("layer0 key/value shapes:", out[0][0].shape, out[0][1].shape)  # expect [8,16,4,64]
