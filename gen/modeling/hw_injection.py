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

        # prototype_keys 是固定的参考点 (来自 JSON)，用于计算相似度权重
        proto = F.normalize(prototype_keys.float(), dim=-1)
        self.register_buffer("prototype_keys", proto)
        num_proto = proto.size(0)
        self.embed_dim = embed_dim

        # prototypes 是可学习的嵌入向量，用于最终的混合输出
        #
        self.prototypes = nn.Parameter(torch.randn(num_proto, embed_dim) * 0.02)

        if trainable_temperature:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))))
            self.register_buffer("temperature", torch.tensor(float(temperature)))
        else:
            self.log_temp = None
            self.register_buffer("temperature", torch.tensor(float(temperature)))

        # 4 个头用于拆分不同硬件语义分量的投影
        self.head_arch = nn.Linear(embed_dim, embed_dim)
        self.head_mem = nn.Linear(embed_dim, embed_dim)
        self.head_cons = nn.Linear(embed_dim, embed_dim)
        self.head_host = nn.Linear(embed_dim, embed_dim)

    def get_temperature(self) -> torch.Tensor:
        if self.log_temp is not None:
            return torch.exp(self.log_temp)
        return self.temperature

    def forward(
        self,
        hw_vec: torch.Tensor,
        return_weights: bool = False,
        split_heads: bool = False,
    ):
        """Map hardware vectors (B, D_key) to embedding vectors.

        Args:
            hw_vec: (B, D_key)
            return_weights: whether to return mixing weights
            split_heads: if True, return shape (B,4,D) via four linear heads
        """

        if hw_vec.ndim != 2:
            raise ValueError("hw_vec must be 2-D (batch, key_dim)")
        proto = self.prototype_keys
        hw_norm = F.normalize(hw_vec, dim=-1)
        
        # 计算输入硬件与固定 key 的相似度
        sim = torch.matmul(hw_norm, proto.t())  # (B, num_proto)
        temp = self.get_temperature()
        weights = torch.softmax(sim / temp, dim=-1)
        
        # 混合可学习的原型向量
        feat = torch.matmul(weights, self.prototypes)  # (B, D)

        if split_heads:
            mix4 = torch.stack(
                [
                    self.head_arch(feat),
                    self.head_mem(feat),
                    self.head_cons(feat),
                    self.head_host(feat),
                ],
                dim=1,
            )  # (B, 4, D)
            if return_weights:
                return mix4, weights
            return mix4
        
        if return_weights:
            return feat, weights
        return feat

    def get_ortho_loss(self) -> torch.Tensor:
        """
        计算可学习原型的正交正则化损失。
        目标：鼓励 self.prototypes 之间尽可能正交（互不相同），防止坍塌。
        """
        # 对原型向量归一化
        p_norm = F.normalize(self.prototypes, dim=-1)
        # 计算 Gram 矩阵 (余弦相似度矩阵)
        gram = torch.matmul(p_norm, p_norm.t())  # (num_proto, num_proto)
        # 目标是单位矩阵 (对角线为1，其余为0)
        eye = torch.eye(gram.size(0), device=gram.device)
        # 计算 MSE 损失
        loss = torch.mean((gram - eye) ** 2)
        return loss

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
