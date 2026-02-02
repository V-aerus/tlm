import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing train_hw_kv_aligner.py when running from repo root
THIS_DIR = Path(__file__).resolve()
GEN_DIR = THIS_DIR.parents[1]
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from train_hw_kv_aligner import SCHEDULE_OP_TOKENS  # noqa: E402


def _normalize_tok(tok: str) -> str:
    tok = tok.strip()
    # Strip common BPE/GPT prefix markers and punctuation
    tok = re.sub(r"^[^A-Za-z0-9]+", "", tok)
    return tok


def build_labels_with_schedule_mask_loose(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Loose schedule mask:
    - convert ids -> tokens
    - normalize token prefix
    - detect first schedule op token
    """
    labels = input_ids.clone()
    sched_set = set(SCHEDULE_OP_TOKENS)
    for b in range(input_ids.size(0)):
        tokens = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in input_ids[b]]
        start = -1
        for i, tok in enumerate(tokens):
            norm = _normalize_tok(tok)
            if norm in sched_set:
                start = i
                break
        if start < 0:
            labels[b, :] = -100
        elif start > 0:
            labels[b, :start] = -100
    return labels


def replace_arch(text: str, arch: str) -> str:
    # Prefer token-level replacement to avoid regex edge cases
    tokens = text.split()
    replaced = False
    for i, t in enumerate(tokens):
        if t.startswith("-arch=sm_"):
            tokens[i] = f"-arch={arch}"
            replaced = True
            break
    if replaced:
        return " ".join(tokens)
    # Fallback to regex in case formatting is unusual
    return re.sub(r"[\\-–—]?arch=sm_\\d+", f"-arch={arch}", text)


def main():
    parser = argparse.ArgumentParser(description="Sanity check: swap -arch in text_full and measure logits delta on schedule tokens.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--arch_a", default="sm_86", help="Original arch to swap from (e.g., sm_86)")
    parser.add_argument("--arch_b", default="sm_70", help="Target arch to swap to (e.g., sm_70)")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_sched_tokens", type=int, default=64)
    parser.add_argument("--debug_samples", type=int, default=2, help="Print debug info when schedule mask is empty")
    parser.add_argument("--debug_scan", type=int, default=0, help="Print first N text_full lines that contain arch_a")
    parser.add_argument("--debug_tokens", type=int, default=0, help="Print token-level schedule detection for first N arch lines")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    found = 0
    debug_left = args.debug_samples
    debug_scan_left = args.debug_scan
    debug_tokens_left = args.debug_tokens
    total = 0
    has_text_full = 0
    has_arch = 0
    repl_ok = 0
    repl_fail = 0
    with open(args.json_path, "r", encoding="utf-8") as f:
        for line in f:
            if found >= args.max_samples:
                break
            obj = json.loads(line)
            text_full = obj.get("text_full")
            total += 1
            if text_full:
                has_text_full += 1
            if text_full and f"-arch={args.arch_a}" in text_full:
                has_arch += 1
            if not text_full or f"-arch={args.arch_a}" not in text_full:
                continue
            if debug_scan_left > 0:
                debug_scan_left -= 1
                print("[DEBUG-SCAN] text_full snippet:", text_full[:300])

            text_a = text_full
            text_b = replace_arch(text_full, args.arch_b)
            if text_a == text_b:
                repl_fail += 1
                if debug_left > 0:
                    debug_left -= 1
                    print("[DEBUG] replace_arch made no change")
                    print(f"  snippet: {text_full[text_full.find('-arch=')-10:text_full.find('-arch=')+30]}")
                continue
            repl_ok += 1

            enc_a = tok(text_a, return_tensors="pt", add_special_tokens=True).to(args.device)
            enc_b = tok(text_b, return_tensors="pt", add_special_tokens=True).to(args.device)

            # Build schedule mask from A (should align with B since only arch token changed)
            labels = build_labels_with_schedule_mask_loose(enc_a["input_ids"], tok)
            mask = (labels != -100)[0]

            if debug_tokens_left > 0:
                debug_tokens_left -= 1
                toks = [tok.convert_ids_to_tokens(int(tid)) for tid in enc_a["input_ids"][0]]
                norm_hits = [(i, t, _normalize_tok(t)) for i, t in enumerate(toks) if _normalize_tok(t) in SCHEDULE_OP_TOKENS]
                print("[DEBUG-TOKENS] schedule detection")
                print(f"  mask_count={int(mask.sum().item())} token_norm_hits={norm_hits[:10]}")
                print(f"  first_tokens={toks[:50]}")

            if mask.sum().item() == 0:
                if debug_left > 0:
                    debug_left -= 1
                    raw_tokens = text_full.split()
                    raw_sched_idx = -1
                    for i, t in enumerate(raw_tokens):
                        if t in SCHEDULE_OP_TOKENS:
                            raw_sched_idx = i
                            break
                    print("[DEBUG] schedule mask empty")
                    print(f"  has_arch_a={f'-arch={args.arch_a}' in text_full} has_arch_b={f'-arch={args.arch_b}' in text_full}")
                    print(f"  raw_sched_idx={raw_sched_idx} raw_sched_tok={raw_tokens[raw_sched_idx] if raw_sched_idx>=0 else 'NA'}")
                    if raw_sched_idx >= 0:
                        lo = max(0, raw_sched_idx - 8)
                        hi = min(len(raw_tokens), raw_sched_idx + 12)
                        print("  raw_sched_window:", " ".join(raw_tokens[lo:hi]))
                    # token-level debug
                    toks = [tok.convert_ids_to_tokens(int(tid)) for tid in enc_a["input_ids"][0]]
                    norm_hits = [(i, t, _normalize_tok(t)) for i, t in enumerate(toks) if _normalize_tok(t) in SCHEDULE_OP_TOKENS]
                    print(f"  token_norm_hits={norm_hits[:10]}")
                    print(f"  first_tokens={toks[:40]}")
                continue

            # Limit to first N schedule tokens
            idx = torch.nonzero(mask, as_tuple=True)[0][: args.max_sched_tokens]

            with torch.no_grad():
                out_a = model(**enc_a)
                out_b = model(**enc_b)
                logits_a = out_a.logits[0][idx]
                logits_b = out_b.logits[0][idx]

                # mean |Δlogits| over vocab
                mean_abs = (logits_a - logits_b).abs().mean().item()

                # KL(softmax(a) || softmax(b)) over vocab
                logp_a = F.log_softmax(logits_a, dim=-1)
                p_a = logp_a.exp()
                logp_b = F.log_softmax(logits_b, dim=-1)
                kl = (p_a * (logp_a - logp_b)).sum(dim=-1).mean().item()

            print(f"[ARCH-SANITY] hw_id={obj.get('hw_id')} hw_name={obj.get('hw_name')}")
            print(f"  arch_a={args.arch_a} arch_b={args.arch_b}")
            print(f"  schedule_tokens={idx.numel()} mean|Δlogits|={mean_abs:.4e} KL={kl:.4e}")
            found += 1

    if found == 0:
        print("No samples with specified -arch found or schedule mask empty.")
        print(f"[DEBUG-SCAN-STAT] total={total} has_text_full={has_text_full} has_arch={has_arch}")
        print(f"[DEBUG-REPL-STAT] repl_ok={repl_ok} repl_fail={repl_fail}")


if __name__ == "__main__":
    main()
