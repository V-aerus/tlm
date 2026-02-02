#!/usr/bin/env python3
"""Router-only calibration across multiple experts (two-stage gating)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from modeling.hw_preprocess import apply_preprocess, summarize_preprocess
from training import compute_gain_loss, entropy_reg, l2r_reg


def _parse_expert_arg(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Invalid --expert '{arg}', expect label=path")
    label, path = arg.split("=", 1)
    return label.strip(), Path(path.strip())


def _parse_kv_arg(arg: str) -> Tuple[str, str]:
    if "=" not in arg:
        raise ValueError(f"Invalid mapping '{arg}', expect key=value")
    key, value = arg.split("=", 1)
    return key.strip(), value.strip()


def _to_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if torch.isnan(torch.tensor(val)):
        return float("nan")
    return val


def _compute_ipct_stats(values: List[float]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    if n % 2 == 1:
        median = vals[n // 2]
    else:
        median = 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    q25 = vals[int(0.25 * (n - 1))]
    q75 = vals[int(0.75 * (n - 1))]
    scale = max(q75 - q25, 1e-3)
    return float(median), float(scale)


def _softplus_inverse(value: float) -> float:
    value_tensor = torch.tensor(value, dtype=torch.float32)
    eps = torch.finfo(value_tensor.dtype).eps
    return torch.log(torch.expm1(torch.clamp(value_tensor, min=eps)))


def _load_preprocess_params(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        params = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(params, dict):
        return None
    if "mean" in params and "std" in params and "mask" in params:
        params.setdefault("source", str(p))
        return params
    return None


class RouterCalibDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        label_to_idx: Dict[str, int],
        *,
        label_field: str = "router_label",
        fallback_label_field: str = "hardware_id",
    ):
        self.samples: List[Tuple[List[float], int, float, float, float]] = []
        self.label_hw_name: Dict[str, str] = {}
        self.ipct_values: List[float] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                label = record.get(label_field) or record.get(fallback_label_field)
                if label not in label_to_idx:
                    continue
                hw_emb = record.get("hw_emb")
                if not hw_emb:
                    continue
                lat_base = _to_float_or_nan(record.get("lat_base_star", record.get("latency")))
                lat_lora = _to_float_or_nan(record.get("lat_lora_star"))
                ipct = float("nan")
                if not torch.isnan(torch.tensor(lat_base)) and not torch.isnan(torch.tensor(lat_lora)):
                    ipct = (lat_base - lat_lora) / (lat_base + 1e-6)
                    self.ipct_values.append(ipct)
                idx = label_to_idx[label]
                self.samples.append((hw_emb, idx, lat_base, lat_lora, ipct))
                hw_name = record.get("hardware_name") or record.get("hardware_id")
                if hw_name and label not in self.label_hw_name:
                    self.label_hw_name[label] = hw_name
        stats = _compute_ipct_stats(self.ipct_values)
        if stats is None:
            self.ipct_center_raw = None
            self.ipct_scale = None
            self.ipct_center = None
        else:
            self.ipct_center_raw, self.ipct_scale = stats
            self.ipct_center = 0.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hw_emb, label_idx, lat_base, lat_lora, ipct = self.samples[idx]
        return {
            "hw_emb": torch.tensor(hw_emb, dtype=torch.float32),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "lat_base": torch.tensor(lat_base, dtype=torch.float32),
            "lat_lora": torch.tensor(lat_lora, dtype=torch.float32),
            "ipct": torch.tensor(ipct, dtype=torch.float32),
        }


@dataclass
class ExpertState:
    label: str
    router_path: Path
    r: torch.nn.Parameter
    s: torch.nn.Parameter
    beta: torch.nn.Parameter
    tau: torch.Tensor
    anchor: Optional[torch.Tensor] = None


def _copy_adapter_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    filenames = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]
    for name in filenames:
        src_path = src / name
        if src_path.exists():
            shutil.copy2(src_path, dst / name)


def _build_anchor_vectors(
    labels: List[str],
    *,
    label_hw_name: Dict[str, str],
    embeddings_path: Path,
    preprocess_params: Optional[Dict],
    anchor_map: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    entries = json.loads(embeddings_path.read_text(encoding="utf-8"))
    emb = {e["hardware_name"]: e["vector"] for e in entries}
    anchors: Dict[str, torch.Tensor] = {}
    for label in labels:
        name = anchor_map.get(label) or label_hw_name.get(label) or label
        vec = emb.get(name)
        if vec is None:
            continue
        p = torch.tensor(vec, dtype=torch.float32)
        if preprocess_params:
            p = apply_preprocess(p, preprocess_params)
        p = F.normalize(p, dim=0)
        anchors[label] = p
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Router-only calibration across multiple experts.")
    parser.add_argument("--dataset-jsonl", required=True, help="Mixed JSONL for router calibration.")
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        help="Expert as label=dir (repeatable). label should match router_label/hardware_id.",
    )
    parser.add_argument("--output-root", required=True, help="Output root dir for calibrated experts.")
    parser.add_argument("--embedding-json", default="gen/Embedding/hardware_embeddings_v4_universe.json")
    parser.add_argument("--preprocess-json", default="gen/Embedding/preprocess_v4u_zscore_v1.json")
    parser.add_argument("--label-field", default="router_label")
    parser.add_argument("--fallback-label-field", default="hardware_id")
    parser.add_argument("--anchor-map", action="append", default=[], help="Optional label=hardware_name mapping.")
    parser.add_argument("--score-mode", default="dot_over_rnorm", choices=["dot", "dot_over_rnorm", "cosine"])
    parser.add_argument("--competition-tau", type=float, default=1.0)
    parser.add_argument("--two-stage", action="store_true", help="Enable two-stage gating (default).")
    parser.add_argument("--no-two-stage", dest="two_stage", action="store_false")
    parser.set_defaults(two_stage=True)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-gain", type=float, default=0.1)
    parser.add_argument("--lambda-g", type=float, default=0.25, help="Strength supervision weight for g* target.")
    parser.add_argument("--lambda-router", type=float, default=1e-4)
    parser.add_argument("--lambda-entropy", type=float, default=0.0)
    parser.add_argument("--lambda-pi-entropy", type=float, default=0.12, help="Entropy regularizer for pi (competition).")
    parser.add_argument("--lambda-anchor", type=float, default=1e-3)
    parser.add_argument("--anchor-mode", default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for competition loss.")
    parser.add_argument("--gain-margin", type=float, default=0.05)
    parser.add_argument("--g-target-scale", type=float, default=0.8, help="Scale multiplier for g* sigmoid slope.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--train-competition-tau",
        action="store_true",
        help="Learn competition_tau during calibration.",
    )
    parser.set_defaults(train_competition_tau=False)
    parser.add_argument("--competition-tau-reg", type=float, default=0.1, help="L2 prior weight for competition_tau.")
    args = parser.parse_args()

    expert_args = [_parse_expert_arg(e) for e in args.expert]
    labels = [label for label, _ in expert_args]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    dataset = RouterCalibDataset(
        Path(args.dataset_jsonl),
        label_to_idx,
        label_field=args.label_field,
        fallback_label_field=args.fallback_label_field,
    )
    if len(dataset) == 0:
        raise RuntimeError("Router calibration dataset is empty after filtering.")
    if dataset.ipct_scale is not None:
        print(
            f"[CONFIG] ipct_stats center_raw={dataset.ipct_center_raw:.4f} "
            f"scale={dataset.ipct_scale:.4f} count={len(dataset.ipct_values)} g_center=0"
        )
    else:
        print("[WARN] ipct_stats unavailable; g* supervision disabled.")

    preprocess_params = _load_preprocess_params(args.preprocess_json)
    if preprocess_params:
        print(f"[CONFIG] router preprocess: {summarize_preprocess(preprocess_params)}")
    else:
        print("[CONFIG] router preprocess: identity")

    anchor_map = dict(_parse_kv_arg(item) for item in args.anchor_map)
    anchors = _build_anchor_vectors(
        labels,
        label_hw_name=dataset.label_hw_name,
        embeddings_path=Path(args.embedding_json),
        preprocess_params=preprocess_params,
        anchor_map=anchor_map,
    )
    missing_anchor = [label for label in labels if label not in anchors]
    if missing_anchor:
        print(f"[WARN] Missing anchor vectors for: {', '.join(missing_anchor)}")

    experts: List[ExpertState] = []
    for label, path in expert_args:
        router_path = path / "router.json"
        if not router_path.exists():
            raise FileNotFoundError(f"router.json not found: {router_path}")
        router_state = json.loads(router_path.read_text(encoding="utf-8"))
        r_vec = torch.tensor(router_state["r"], dtype=torch.float32, device=args.device)
        s_val = router_state.get("s")
        if s_val is None:
            s_val = float(torch.linalg.norm(r_vec).item())
        s_val = max(float(s_val), 1e-6)
        beta_val = float(router_state.get("beta", -1.0))
        tau_val = float(router_state.get("tau", 2.0))
        expert = ExpertState(
            label=label,
            router_path=router_path,
            r=torch.nn.Parameter(r_vec),
            s=torch.nn.Parameter(torch.tensor(_softplus_inverse(s_val), dtype=torch.float32, device=args.device)),
            beta=torch.nn.Parameter(torch.tensor(beta_val, dtype=torch.float32, device=args.device)),
            tau=torch.tensor(tau_val, dtype=torch.float32, device=args.device),
            anchor=anchors.get(label),
        )
        experts.append(expert)

    r_params = torch.nn.ParameterList([e.r for e in experts])
    s_params = torch.nn.ParameterList([e.s for e in experts])
    beta_params = torch.nn.ParameterList([e.beta for e in experts])
    comp_tau_param = None
    if args.train_competition_tau:
        init_tau = max(float(args.competition_tau), 1e-6)
        comp_tau_param = torch.nn.Parameter(
            torch.tensor(_softplus_inverse(init_tau), dtype=torch.float32, device=args.device)
        )
    opt_params = list(r_params) + list(s_params) + list(beta_params)
    if comp_tau_param is not None:
        opt_params.append(comp_tau_param)
    optimizer = torch.optim.AdamW(opt_params, lr=args.learning_rate)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            hw_raw = batch["hw_emb"].to(args.device)
            labels = batch["label"].to(args.device)
            lat_base = batch["lat_base"].to(args.device)
            lat_lora = batch["lat_lora"].to(args.device)

            h = apply_preprocess(hw_raw, preprocess_params) if preprocess_params else hw_raw

            r_stack = torch.stack([e.r for e in experts], dim=0)  # [E, D]
            r_dir = F.normalize(r_stack, dim=1, eps=1e-6)
            s_vec = F.softplus(torch.stack([e.s for e in experts], dim=0))  # [E]
            dot = h @ r_dir.t()  # [B, E]
            b = -F.softplus(torch.stack([e.beta for e in experts], dim=0))  # [E]
            tau = torch.stack([e.tau for e in experts], dim=0)  # [E]

            # Stage A: strength gate
            l = (dot * s_vec + b) / tau
            g = torch.sigmoid(l)

            # Stage B: competition
            if args.score_mode == "dot":
                u = dot
            elif args.score_mode == "cosine":
                h_norm = torch.linalg.norm(h, dim=1, keepdim=True)
                u = dot / (h_norm + 1e-6)
            else:
                u = dot
            if comp_tau_param is not None:
                comp_tau = F.softplus(comp_tau_param) + 1e-6
            else:
                comp_tau = torch.tensor(float(args.competition_tau), device=args.device)
            u = u / torch.clamp(comp_tau, min=1e-6)
            pi = torch.softmax(u, dim=1)

            log_pi = torch.log(pi + 1e-8)
            if args.lambda_cls > 0:
                if args.label_smoothing > 0:
                    cls_loss = args.lambda_cls * F.cross_entropy(
                        u, labels, label_smoothing=args.label_smoothing
                    )
                else:
                    cls_loss = args.lambda_cls * F.nll_loss(log_pi, labels)
            else:
                cls_loss = torch.zeros((), device=args.device)

            pi_entropy = -(pi * log_pi).sum(dim=1).mean()
            pi_entropy_loss = -args.lambda_pi_entropy * pi_entropy if args.lambda_pi_entropy > 0 else torch.zeros(
                (), device=args.device
            )

            # Gain loss (target expert only)
            paired_mask = (~torch.isnan(lat_base)) & (~torch.isnan(lat_lora))
            if paired_mask.any():
                lat_base_p = lat_base[paired_mask]
                lat_lora_p = lat_lora[paired_mask]
                labels_p = labels[paired_mask]
                g_label = g[paired_mask, :].gather(1, labels_p.unsqueeze(1)).squeeze(1)
                I_pct = (lat_base_p - lat_lora_p) / (lat_base_p + 1e-6)
                gain_loss = compute_gain_loss(
                    I_pct=I_pct,
                    g_mean=g_label.mean(),
                    step=global_step,
                    warmup_steps=args.warmup_steps,
                    m_target=args.gain_margin,
                    lambda_gain=args.lambda_gain,
                )
                entropy_loss = entropy_reg(g_label, args.lambda_entropy)
            else:
                gain_loss = torch.zeros((), device=args.device)
                entropy_loss = torch.zeros((), device=args.device)

            g_target_loss = torch.zeros((), device=args.device)
            g_target_mean = torch.tensor(0.0, device=args.device)
            g_pred_mean = torch.tensor(0.0, device=args.device)
            if args.lambda_g > 0 and dataset.ipct_scale is not None:
                ipct = batch["ipct"].to(args.device)
                ipct_mask = ~torch.isnan(ipct)
                if ipct_mask.any():
                    g_pred = g.gather(1, labels.unsqueeze(1)).squeeze(1)
                    scale = max(dataset.ipct_scale * args.g_target_scale, 1e-3)
                    g_target = torch.sigmoid(ipct / scale)
                    g_target = g_target[ipct_mask]
                    g_pred = g_pred[ipct_mask]
                    g_target_loss = args.lambda_g * F.mse_loss(g_pred, g_target)
                    g_target_mean = g_target.mean()
                    g_pred_mean = g_pred.mean()

            router_reg = l2r_reg(r_stack, args.lambda_router)

            anchor_loss = torch.zeros((), device=args.device)
            if args.lambda_anchor > 0 and any(e.anchor is not None for e in experts):
                terms = []
                for e in experts:
                    if e.anchor is None:
                        continue
                    r_dir_e = F.normalize(e.r, dim=0, eps=1e-6)
                    if args.anchor_mode == "l2":
                        diff = r_dir_e - e.anchor.to(args.device)
                        terms.append(diff.pow(2).mean())
                    else:
                        cos = (r_dir_e * e.anchor.to(args.device)).sum()
                        terms.append(1.0 - cos)
                if terms:
                    anchor_loss = args.lambda_anchor * torch.stack(terms).mean()

            tau_reg = torch.zeros((), device=args.device)
            if comp_tau_param is not None and args.competition_tau_reg > 0:
                tau_reg = args.competition_tau_reg * (comp_tau - float(args.competition_tau)).pow(2)

            loss = cls_loss + gain_loss + entropy_loss + g_target_loss + pi_entropy_loss + router_reg + anchor_loss + tau_reg
            loss.backward()
            optimizer.step()

            if global_step % 50 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                    f"gain={gain_loss.item():.4f} g_target={g_target_loss.item():.4f} "
                    f"g_t_mean={g_target_mean.item():.4f} g_p_mean={g_pred_mean.item():.4f} "
                    f"pi_ent={pi_entropy_loss.item():.4f} anchor={anchor_loss.item():.4f} "
                    f"router={router_reg.item():.4f}"
                )
            global_step += 1

    comp_tau_value = float(args.competition_tau)
    if comp_tau_param is not None:
        comp_tau_value = float(F.softplus(comp_tau_param).detach().cpu().item())

    # Save calibrated routers
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for expert in experts:
        src_dir = expert.router_path.parent
        out_dir = out_root / expert.label
        _copy_adapter_files(src_dir, out_dir)

        router_state = json.loads(expert.router_path.read_text(encoding="utf-8"))
        meta = router_state.get("meta", {})
        if preprocess_params:
            meta["preprocess"] = preprocess_params
        meta["score_mode"] = args.score_mode
        meta["two_stage"] = bool(args.two_stage)
        meta["competition_tau"] = comp_tau_value
        meta["train_competition_tau"] = bool(args.train_competition_tau)
        meta["competition_tau_reg"] = float(args.competition_tau_reg)
        meta["anchor_mode"] = args.anchor_mode
        meta["anchor_lambda"] = float(args.lambda_anchor)
        meta["calib_dataset"] = args.dataset_jsonl
        meta["calib_labels"] = list(label_to_idx.keys())
        meta["lambda_g"] = float(args.lambda_g)
        meta["lambda_pi_entropy"] = float(args.lambda_pi_entropy)
        meta["label_smoothing"] = float(args.label_smoothing)
        meta["router_param"] = "s_norm"
        if dataset.ipct_scale is not None:
            meta["g_target_center"] = 0.0
            if dataset.ipct_center_raw is not None:
                meta["g_target_center_raw"] = float(dataset.ipct_center_raw)
            meta["g_target_scale"] = float(dataset.ipct_scale * args.g_target_scale)

        router_state.update(
            {
                "hardware_dim": int(expert.r.numel()),
                "r": expert.r.detach().cpu().tolist(),
                "s": float(F.softplus(expert.s).detach().cpu().item()),
                "beta": float(expert.beta.detach().cpu()),
                "tau": float(expert.tau.detach().cpu()),
                "lambda_router": args.lambda_router,
                "lambda_entropy": args.lambda_entropy,
                "lambda_gain": args.lambda_gain,
                "meta": meta,
            }
        )
        out_path = out_dir / "router.json"
        out_path.write_text(json.dumps(router_state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] {expert.label} -> {out_path}")


if __name__ == "__main__":
    main()
