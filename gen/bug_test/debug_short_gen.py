#!/usr/bin/env python3
# 调试脚本：在训练样本上短程生成，检查语法是否可被 TVM State 解析

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling.hw_injection import ProtoMixAligner, build_prototype_matrix
from peft import PeftModel


def find_subsequence(seq, pattern):
    n, m = len(seq), len(pattern)
    for i in range(n - m + 1):
        if seq[i : i + m] == pattern:
            return i
    return -1


def main():
    parser = argparse.ArgumentParser(description="短程生成调试：可选 regressor，对比单 MASK 或直接 teacher prompt。")
    parser.add_argument("--data_path", default="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/align_train_multi_merged_with_number_4mask/0_merge.json")
    parser.add_argument("--model_path", default="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1")
    parser.add_argument("--adapter_path", default="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/hw_aligner_lora_proto_with_number_4mask")
    parser.add_argument("--aligner_path", default=None, help="ProtoMix 对齐器 ckpt，默认 adapter_path/hw_aligner_lora.pt")
    parser.add_argument("--regressor_path", default=None, help="回归器 ckpt，若指定则仅回归 hw_emb，不走对齐器")
    parser.add_argument("--emb_path", default="/home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v3.json")
    parser.add_argument("--hw_name_filter", default="nvidia/nvidia-a40")
    parser.add_argument("--gen_tokens", type=int, default=50)
    parser.add_argument("--use_teacher_prompt", action="store_true", help="使用原始 text（含 target），不替换硬件段")
    parser.add_argument("--regress_teacher_target", action="store_true", help="仅在 use_teacher_prompt 时有效：用 regressor 替换原始 target 的 embedding 段")
    args = parser.parse_args()

    # 路径配置
    data_path = args.data_path
    model_path = args.model_path
    adapter_path = args.adapter_path
    aligner_path = args.aligner_path or f"{adapter_path}/hw_aligner_lora.pt"
    regressor_path = args.regressor_path
    emb_path = args.emb_path

    proto_names = [
        "nvidia/nvidia-v100",
        "nvidia/nvidia-a40",
        "nvidia/jetson-agx-xavier",
        "aws/cpu/c5.18xlarge",
    ]
    hw_name_filter = args.hw_name_filter
    gen_tokens = args.gen_tokens

    # 1) 取一条样本
    sample = None
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("hw_name") == hw_name_filter:
                sample = obj
                break
    if sample is None:
        raise ValueError(f"No sample found for hw_name={hw_name_filter}")
    print(f"Loaded sample with hw_name={sample['hw_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 加载模型/LoRA
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path)
    base.eval().to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval().to(device)
    embed_layer = model.get_input_embeddings()

    # 3) 加载对齐器/硬件向量（或回归器）
    emb_dict = {e["hardware_name"]: e["vector"] for e in json.load(open(emb_path))}
    hw_vec = torch.tensor(sample["hw_emb"], device=device).unsqueeze(0)

    if regressor_path:
        # 仅做几何回归：hw_emb -> target embedding (均值)，不走 ProtoMix/LoRA 注入
        ck = torch.load(regressor_path, map_location="cpu")
        in_dim, out_dim = ck["in_dim"], ck["out_dim"]
        hidden = ck.get("hidden", 256)
        reg = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        ).to(device)
        state = ck["state_dict"]
        # 兼容保存时的 "net." 前缀
        if any(k.startswith("net.") for k in state.keys()):
            state = {k.replace("net.", ""): v for k, v in state.items()}
        reg.load_state_dict(state, strict=False)
        reg.eval()
        with torch.no_grad():
            hw_embed_single = reg(hw_vec)  # (1, D)
    else:
        proto_mat = build_prototype_matrix(emb_dict, proto_names)
        aligner = ProtoMixAligner(proto_mat, embed_dim=embed_layer.embedding_dim, temperature=1.0, trainable_temperature=False)
        ckpt = torch.load(aligner_path, map_location="cpu")
        aligner.load_state_dict(ckpt["state_dict"])
        aligner.eval().to(device)
        hw_embed_single = aligner(hw_vec)  # (1,D)

    # 4) 构造前缀（teacher prompt 或 student prompt）
    if args.use_teacher_prompt:
        prompt_text = sample["text"]
        enc = tok(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"][:, :-1].to(device)  # 去掉最后一 token 以便生成
        attn_mask = enc["attention_mask"][:, :-1].to(device)
        embeds = embed_layer(input_ids)

        if args.regress_teacher_target:
            # 使用 student prompt 的 MASK 位置作为替换参考，在 teacher prompt 对应位置插入回归向量
            stu_ids = tok(sample["text_student"], add_special_tokens=False)["input_ids"]
            mask_id = tok("[MASK]", add_special_tokens=False)["input_ids"][0]
            if mask_id in stu_ids:
                mask_pos = stu_ids.index(mask_id)
                if mask_pos < embeds.size(1):
                    embeds = embeds.clone()
                    embeds[:, mask_pos, :] = hw_embed_single
            else:
                print("[WARN] student prompt has no MASK; skip regress_teacher_target replacement")
    else:
        hw_token_ids = tok("[MASK]", add_special_tokens=False)["input_ids"]
        enc = tok(sample["text_student"], return_tensors="pt")
        input_ids = enc["input_ids"][:, :-1].to(device)  # 去掉最后一 token 以便生成
        attn_mask = enc["attention_mask"][:, :-1].to(device)
        pos = find_subsequence(input_ids[0].tolist(), hw_token_ids)
        if pos < 0:
            raise ValueError("MASK span not found in text_student")
        embeds = embed_layer(input_ids)
        embeds[:, pos, :] = hw_embed_single

    # 5) 短程生成
    with torch.no_grad():
        prompt_len = input_ids.shape[-1]
        gen_ids = model.generate(
            input_ids=input_ids,  # 传入原始 token 以确保长度对齐
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            max_new_tokens=gen_tokens,
            min_new_tokens=1,
            do_sample=False,
            num_beams=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = gen_ids[:, prompt_len :][0].tolist()
    decoded_tail = tok.decode(new_tokens)

    print(f"Generated token count: {len(new_tokens)}")
    if len(new_tokens) == 0:
        print(f"[WARN] No new tokens generated. prompt_len={prompt_len}, max_len={max_len}, max_new_tokens={gen_tokens}")
    print("Generated tail:\n", decoded_tail if decoded_tail.strip() else "<EMPTY>")

    # 6) 打印组合后的文本供人工检查
    prefix = tok.decode(input_ids[0])
    full = prefix + tok.decode(new_tokens)
    print("\nFull text (prefix + generated):\n", full)


if __name__ == "__main__":
    main()
