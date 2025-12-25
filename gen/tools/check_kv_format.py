import torch
from transformers import AutoModelForCausalLM, AutoConfig
from gen.hw_kv_aligner import HwKVAligner

def test_kv_compatibility():
    # 1. 模拟一个非常小的模型配置
    config = AutoConfig.from_pretrained("gpt2") # 或你的目标模型类型
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 32
    
    # 2. 实例化模型和你的 Aligner
    print("Loading dummy model...")
    model = AutoModelForCausalLM.from_config(config)
    aligner = HwKVAligner(
        llm_num_layers=config.n_layer,
        llm_num_heads=config.n_head,
        llm_head_dim=config.n_embd // config.n_head,
        hw_dim=24
    )

    # 3. 构造 Dummy 输入
    B = 1
    hw_emb = torch.randn(B, 24)
    input_ids = torch.tensor([[101, 102]]) # dummy tokens

    # 4. 获取 Aligner 的输出
    print("Running Aligner...")
    past_key_values = aligner(hw_emb, batch_size=B)
    
    print(f"Aligner output type: {type(past_key_values)}")
    print(f"Layer 0 type: {type(past_key_values[0])}")
    if isinstance(past_key_values[0], tuple):
        print(f"Layer 0 is Tuple, key shape: {past_key_values[0][0].shape}, value shape: {past_key_values[0][1].shape}")

    # 5. 关键测试：尝试喂给模型
    print("-" * 30)
    print("Attempting model.forward with past_key_values...")
    try:
        # 尝试调用模型生成，看是否报错
        with torch.no_grad():
            # 注意：generate 内部逻辑更复杂，我们先测 forward 这一关
            outputs = model(input_ids=input_ids, past_key_values=past_key_values)
        print("✅ Success! Model accepted this format.")
    except Exception as e:
        print("❌ Failed! Model rejected this format.")
        print(f"Error message: {e}")
        print("\n建议：请修改 hw_kv_aligner.py，返回 (key, value) 元组格式。")

if __name__ == "__main__":
    test_kv_compatibility()
