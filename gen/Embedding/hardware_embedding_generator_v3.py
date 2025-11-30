#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件特征向量生成器 V3
目标：为 4-Token 注入方案提供更丰富的“超级向量”，显式编码架构、内存/约束、宿主信息。
生成结果写入 hardware_embeddings_v3.json。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


# 基础类别定义
VENDOR_CATS = ["nvidia", "intel", "arm", "amd"]
HOST_ARCH_CATS = ["x86", "aarch64"]


@dataclass
class HardwareAttr:
    name: str
    vendor: str
    compute_cap: float          # 架构/Compute Capability，例：8.6
    shared_mem_kb: int
    registers_per_block: int
    l2_cache_kb: int
    cons: List[int]             # 那 8 个整数（保持原顺序）
    host_cores: int = 0
    host_arch: str = None       # x86 / aarch64 / None 表示未知或无宿主


# 覆盖当前实验涉及的 4 种硬件
HARDWARE_ATTRS: Dict[str, HardwareAttr] = {
    "nvidia/nvidia-v100": HardwareAttr(
        name="nvidia/nvidia-v100",
        vendor="nvidia",
        compute_cap=7.0,
        shared_mem_kb=49152,
        registers_per_block=65536,
        l2_cache_kb=6144,
        cons=[-1, 16, 64, 49152, 12345678, 1024, 8, 32],
    ),
    "nvidia/nvidia-a40": HardwareAttr(
        name="nvidia/nvidia-a40",
        vendor="nvidia",
        compute_cap=8.6,
        shared_mem_kb=49152,
        registers_per_block=65536,
        l2_cache_kb=6144,
        cons=[-1, 16, 64, 49152, 12345678, 1024, 8, 32],
    ),
    "nvidia/jetson-agx-xavier": HardwareAttr(
        name="nvidia/jetson-agx-xavier",
        vendor="nvidia",
        compute_cap=7.2,
        shared_mem_kb=49152,
        registers_per_block=65536,
        l2_cache_kb=512,
        cons=[-1, 16, 64, 49152, 12345678, 1024, 8, 32],
        host_cores=8,
        host_arch="aarch64",
    ),
    "aws/cpu/c5.18xlarge": HardwareAttr(
        name="aws/cpu/c5.18xlarge",
        vendor="intel",
        compute_cap=5.0,  # 近似占位，用于区分 CPU 代际
        shared_mem_kb=0,
        registers_per_block=0,
        l2_cache_kb=25600,  # 25MB 近似值（按 socket）
        cons=[36, 64, 64, 0, 0, 0, 0, 0],  # 对应数据集中的 8 整数
        host_cores=36,
        host_arch="x86",
    ),
}


def one_hot(index: int, length: int) -> List[float]:
    vec = [0.0] * length
    if 0 <= index < length:
        vec[index] = 1.0
    return vec


def build_normalizers(attrs: Dict[str, HardwareAttr]) -> Dict[str, List[float]]:
    """根据已知硬件求各段的归一化系数。"""
    shared_max = max(a.shared_mem_kb for a in attrs.values())
    regs_max = max(a.registers_per_block for a in attrs.values())
    l2_max = max(a.l2_cache_kb for a in attrs.values())
    host_core_max = max(max(a.host_cores for a in attrs.values()), 1)

    # 逐位置的 cons 最大值，避免除以 0
    cons_len = max(len(a.cons) for a in attrs.values())
    cons_max = [1] * cons_len
    for a in attrs.values():
        for i, v in enumerate(a.cons):
            cons_max[i] = max(cons_max[i], abs(v))

    return {
        "shared_max": float(shared_max or 1),
        "regs_max": float(regs_max or 1),
        "l2_max": float(l2_max or 1),
        "host_core_max": float(host_core_max or 1),
        "cons_max": cons_max,
    }


def build_vector(attr: HardwareAttr, norm: Dict[str, List[float]]) -> List[float]:
    # Arch 段：vendor one-hot + compute_cap 归一到 0~1（粗略除以 10）
    vendor_idx = VENDOR_CATS.index(attr.vendor) if attr.vendor in VENDOR_CATS else -1
    arch_vec = one_hot(vendor_idx, len(VENDOR_CATS)) + [attr.compute_cap / 10.0]

    # Mem 段：共享内存 / 寄存器 / L2
    mem_vec = [
        attr.shared_mem_kb / norm["shared_max"],
        attr.registers_per_block / norm["regs_max"],
        attr.l2_cache_kb / norm["l2_max"],
    ]

    # Cons 段：8 个整数，逐位置归一
    cons_vec = []
    for i, vmax in enumerate(norm["cons_max"]):
        v = attr.cons[i] if i < len(attr.cons) else 0
        cons_vec.append(v / vmax if vmax != 0 else 0.0)

    # Host 段：host cores + host arch one-hot
    host_arch_idx = HOST_ARCH_CATS.index(attr.host_arch) if attr.host_arch in HOST_ARCH_CATS else -1
    host_vec = [
        attr.host_cores / norm["host_core_max"],
        *one_hot(host_arch_idx, len(HOST_ARCH_CATS)),
    ]

    return arch_vec + mem_vec + cons_vec + host_vec


def main() -> None:
    norm = build_normalizers(HARDWARE_ATTRS)
    vectors = []
    for name, attr in HARDWARE_ATTRS.items():
        vec = build_vector(attr, norm)
        vectors.append({"hardware_name": name, "vector": vec})

    out_path = Path(__file__).with_name("hardware_embeddings_v3.json")
    out_path.write_text(json.dumps(vectors, indent=2, ensure_ascii=False))
    print(f"Saved {len(vectors)} vectors to {out_path}")


if __name__ == "__main__":
    main()
