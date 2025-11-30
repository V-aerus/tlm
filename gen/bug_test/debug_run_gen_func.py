"""
脱离 TVM，直接调用 gen_func 做一次最小化生成测试。
使用方法：
  CUDA_VISIBLE_DEVICES=3 python gen/debug_run_gen_func.py \
    --model_path /path/to/clm_gen_multi_v1 \
    --sketch_path /path/to/0_merge.json \
    --hw_aligner_path /path/to/hw_aligner_lora.pt \
    --edge_embedding_path /path/to/hardware_embeddings_v3.json \
    --hw_token "[MASK] [MASK] [MASK] [MASK]" \
    --hw_name "nvidia/nvidia-a40" \
    --max_new_tokens 32
"""
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tvm import auto_scheduler

from gen_state_debug_regressor import gen_func, prepare_hw_injection_context
from make_dataset import input_to_tokens
from common import register_data_path, load_and_register_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sketch_path", required=True)
    parser.add_argument("--hw_aligner_path", default=None)
    parser.add_argument("--hw_regressor_path", default=None)
    parser.add_argument("--edge_embedding_path", required=True)
    parser.add_argument("--hw_token", default="[MASK]")
    parser.add_argument("--hw_name", required=True, help="例如 nvidia/nvidia-a40")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", required=False, default=None, help="用于注册 workload 的 target 字符串，可与 gen_state 相同")
    args = parser.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()

    # 确保 TVM workload 已注册（与 gen_state 保持一致）
    if args.target:
        register_data_path(args.target)
    load_and_register_tasks()

    # 取一条 MeasureInput 和 state
    inputs, _ = auto_scheduler.RecordReader(args.sketch_path).read_lines()
    inp = auto_scheduler.measure.recover_measure_input(inputs[0], rebuild_state=True)
    task = inp.task
    states = [inp.state]
    print(f"[INFO] Loaded one workload: {task.workload_key}")

    # 读取 hw embedding
    with open(args.edge_embedding_path, "r", encoding="utf-8") as f:
        emb_json = json.load(f)
    emb_map = {e["hardware_name"]: e["vector"] for e in emb_json}
    if args.hw_name not in emb_map:
        raise KeyError(f"hw_name {args.hw_name} not found in embedding file")
    hw_vec = emb_map[args.hw_name]

    # 构造注入上下文
    hw_cfg = {
        "aligner_path": args.hw_aligner_path,
        "hw_regressor_path": args.hw_regressor_path,
        "hw_vec": hw_vec,
        "hw_token": args.hw_token,
        "hw_name": args.hw_name,
    }
    ctx = prepare_hw_injection_context(hw_cfg, tok, model, device)

    # 构造 prompt 文本
    tokens = input_to_tokens(task, states, inp, hw_token_placeholder=ctx["placeholder"])
    print("[PROMPT]", tokens[0])

    gens = gen_func(
        task,
        states,
        inp,
        tok,
        model,
        device,
        {"max_new_tokens": args.max_new_tokens},
        hw_injection_ctx=ctx,
    )
    print("[GEN TOKENS]", gens)


if __name__ == "__main__":
    main()
