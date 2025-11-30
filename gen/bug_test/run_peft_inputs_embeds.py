"""
实验 D：验证 PeftModel 在 inputs_embeds 路径下的 generate 行为。

运行示例：
CUDA_VISIBLE_DEVICES=3 python gen/bug_test/run_peft_inputs_embeds.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
  --adapter_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/hw_aligner_lora_stage1 \
  --prompt "p0 p1 T_batch_matmul_NN 3ed6b8b696d74a428d188b1a05553246 1 128 3072 1 3072 768 1 128 768 [MASK] [MASK] [MASK] [MASK]"
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True, help="LoRA/PEFT 适配器目录，包含 adapter_model.safetensors")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if PeftModel is None:
        raise ImportError("peft 未安装，无法加载 PeftModel")

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()
    peft = PeftModel.from_pretrained(base, args.adapter_path).to(device).eval()

    enc = tok(args.prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        out1 = base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        out2 = peft.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        embeds = peft.get_input_embeddings()(input_ids)
        out3 = peft.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    def report(name, out):
        gen_part = out[:, prompt_len:]
        print(f"{name}: out_shape={list(out.shape)}, prompt_len={prompt_len}, gen_len={gen_part.shape[-1]}")
        print(f"{name}: gen_ids={gen_part[0].tolist() if gen_part.numel() else []}")
        print(f"{name}: gen_text='{tok.decode(gen_part[0], skip_special_tokens=False)}'")

    report("base_input_ids", out1)
    report("peft_input_ids", out2)
    report("peft_inputs_embeds", out3)


if __name__ == "__main__":
    main()
