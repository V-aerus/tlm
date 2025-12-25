任务 1：让 HF generate 的每一步都吃到你想要的 position_ids（核心修复）

给 codex 的指令可以这么写（重点是 patch prepare_inputs_for_generation，每步塞 position_ids）：

在 KV 模式调用 model.generate() 之前，monkey-patch：

用 attention_mask 计算 position_ids

计算 position 时强制忽略 prefix_len（即使 prefix_mask=1 也要忽略它对 position 的贡献）

每一步只取当前输入长度那一段：position_ids[:, -input_ids.shape[1]:]

伪代码要点（让 codex照这个写进 gen_state_debug_kv.py）：

orig_prepare = model.prepare_inputs_for_generation
prefix_len = hw_past[0][0].shape[-2]  # or args.hw_kv_num_slots

def patched_prepare(input_ids, past_key_values=None, attention_mask=None, **kwargs):
    model_inputs = orig_prepare(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        **kwargs
    )
    if attention_mask is not None:
        pos_mask = attention_mask.long()
        # ignore prefix slots for position counting
        if prefix_len > 0 and pos_mask.shape[1] >= prefix_len:
            pos_mask[:, :prefix_len] = 0
        position_ids = pos_mask.cumsum(-1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        position_ids = position_ids[:, -input_ids.shape[1]:]
        model_inputs["position_ids"] = position_ids
    return model_inputs

model.prepare_inputs_for_generation = patched_prepare


同时保留你现在的 forward 打点：你应该能看到 generate loop 里 不再是 position_ids=None。