#!/usr/bin/env python3
"""
快速验证：用 hw_embed regressor 替换原始硬件 target 段的 embedding，再生成张量句子。

用途：
- 绕过 ProtoMix/LoRA，仅做几何对齐，查看在“原始 prompt”下生成的句子是否更接近合法语法。
- 不涉及 TVM 运行，只生成文本供人工/后续脚本检查；你可将输出再喂给现有 parser/TVM 链路验证合法性。

用法示例：
python gen/debug_regressor_gen.py \
  --data_path /home/.../align_train_multi_merged_with_number_4mask/0_merge.json \
  --model_path /home/.../clm_gen_multi_v1 \
  --adapter_path /home/.../hw_aligner_lora_proto_with_number_4mask \
  --regressor_path /home/.../hw_embed_regressor.pt \
  --hw_name_filter nvidia/nvidia-a40 \
  --gen_tokens 50
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--regressor_path", required=True, help="hw_embed regressor ckpt")
    parser.add_argument("--hw_name_filter", default="nvidia/nvidia-a40")
    parser.add_argument("--gen_tokens", type=int, default=50)
    args = parser.parse_args()

    # 取一条样本
    sample = None
    with open(args.data_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("hw_name") == args.hw_name_filter:
                sample = obj
                break
    if sample is None:
        raise ValueError(f"No sample found for hw_name={args.hw_name_filter}")
    print(f"Loaded sample with hw_name={sample['hw_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型/LoRA
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter_path).to(device).eval()
    embed_layer = model.get_input_embeddings()

    # 加载 regressor
    ck = torch.load(args.regressor_path, map_location="cpu")
    in_dim, out_dim = ck["in_dim"], ck["out_dim"]
    hidden = ck.get("hidden", 256)
    reg = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_dim),
    ).to(device)
    state = ck["state_dict"]
    if any(k.startswith("net.") for k in state.keys()):
        state = {k.replace("net.", ""): v for k, v in state.items()}
    reg.load_state_dict(state, strict=False)
    reg.eval()

    # 构造 teacher prompt（原始 text），用 regressor 替换硬件段的 embedding（基于 student MASK 位置）
    enc_teacher = tok(sample["text"], return_tensors="pt")
    input_ids = enc_teacher["input_ids"][:, :-1].to(device)  # 去掉最后一 token 生成
    attn_mask = enc_teacher["attention_mask"][:, :-1].to(device)
    embeds = embed_layer(input_ids)

    # 用 student prompt 中的 MASK 位置作为参考，将该位置 embedding 替换为 regressor 输出
    student_ids = tok(sample["text_student"], add_special_tokens=False)["input_ids"]
    mask_id = tok("[MASK]", add_special_tokens=False)["input_ids"][0]
    if mask_id in student_ids:
        mask_pos = student_ids.index(mask_id)
        hw_vec = torch.tensor(sample["hw_emb"], device=device).unsqueeze(0)
        with torch.no_grad():
            hw_embed = reg(hw_vec)  # (1,D)
        if mask_pos < embeds.size(1):
            embeds = embeds.clone()
            embeds[:, mask_pos, :] = hw_embed
        else:
            print(f"[WARN] mask_pos {mask_pos} exceeds prompt length {embeds.size(1)}; skip replacement")
    else:
        print("[WARN] student prompt has no [MASK]; skip replacement")

    # 生成
    with torch.no_grad():
        prompt_len = input_ids.shape[-1]
        gen_ids = model.generate(
            input_ids=input_ids,
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            max_new_tokens=args.gen_tokens,
            min_new_tokens=1,
            do_sample=False,
            num_beams=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = gen_ids[:, prompt_len:][0].tolist()
    decoded_tail = tok.decode(new_tokens)

    print(f"Generated token count: {len(new_tokens)}")
    print("Generated tail:\n", decoded_tail if decoded_tail.strip() else "<EMPTY>")
    prefix = tok.decode(input_ids[0])
    print("\nFull text (prefix + generated):\n", prefix + decoded_tail)


if __name__ == "__main__":
    main()
