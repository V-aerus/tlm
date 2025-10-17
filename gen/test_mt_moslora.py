#!/usr/bin/env python3
"""
MT-MoSLoRA测试脚本
测试HA+HS双轨制架构的功能
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_mt_moslora import MTMoSLoRALinear, apply_mt_moslora_to_model, ModelArguments


def test_mt_moslora_linear():
    """测试MTMoSLoRALinear的基本功能"""
    print("=" * 60)
    print("Testing MTMoSLoRALinear Basic Functionality")
    print("=" * 60)
    
    try:
        # 创建配置
        ha_config = {
            'r': 8,
            'alpha': 16,
            'dropout': 0.1,
            'use_mixer': True
        }
        
        hs_config = {
            'r': 8,
            'alpha': 16,
            'dropout': 0.1,
            'use_mixer': True
        }
        
        hardware_types = ['v100', 'xavier', 'i7']
        
        # 创建MT-MoSLoRA模块
        module = MTMoSLoRALinear(
            in_features=768,
            out_features=768,
            ha_config=ha_config,
            hs_config=hs_config,
            hardware_types=hardware_types
        )
        
        # 测试输入
        batch_size, seq_len, hidden_size = 2, 64, 768
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # 测试不同硬件的前向传播
        test_hardwares = ['v100', 'xavier', 'i7', 'unknown']
        
        for hw in test_hardwares:
            output = module(x, hardware_id=hw)
            
            # 验证输出形状
            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
            
            print(f"Hardware: {hw}")
            print(f"  Routed to: {module.route_hardware(hw)}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output norm: {output.norm().item():.4f}")
        
        # 获取参数统计
        stats = module.get_trainable_parameters()
        print(f"\nParameter Statistics:")
        print(f"  HA params: {stats['ha_params']:,}")
        print(f"  HS params: {stats['hs_params']:,}")
        print(f"  Total params: {stats['total_params']:,}")
        print(f"  Number of HS experts: {stats['num_hs_experts']}")
        
        print("✅ MTMoSLoRALinear test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MTMoSLoRALinear test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_routing():
    """测试硬件路由功能"""
    print("\n" + "=" * 60)
    print("Testing Hardware Routing")
    print("=" * 60)
    
    try:
        ha_config = {'r': 4, 'alpha': 8, 'dropout': 0.1, 'use_mixer': True}
        hs_config = {'r': 4, 'alpha': 8, 'dropout': 0.1, 'use_mixer': True}
        hardware_types = ['v100', 'xavier', 'i7']
        
        module = MTMoSLoRALinear(
            in_features=512,
            out_features=512,
            ha_config=ha_config,
            hs_config=hs_config,
            hardware_types=hardware_types
        )
        
        # 测试硬件路由
        test_cases = [
            ("nvidia/nvidia-v100", "v100"),
            ("nvidia/jetson-agx-xavier", "xavier"),
            ("intel/i7", "i7"),
            ("v100", "v100"),
            ("xavier", "xavier"),
            ("i7", "i7"),
            ("unknown/hardware", "v100"),  # 默认路由
        ]
        
        for hardware, expected_type in test_cases:
            routed_type = module.route_hardware(hardware)
            print(f"Hardware: {hardware} -> Routed to: {routed_type}")
            assert routed_type == expected_type, f"Expected {expected_type}, got {routed_type}"
        
        print("✅ Hardware routing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hardware routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_integration():
    """测试模型集成"""
    print("\n" + "=" * 60)
    print("Testing Model Integration")
    print("=" * 60)
    
    try:
        # 创建一个简单的GPT-2模型用于测试
        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=256,
            n_layer=2,
            n_head=4
        )
        
        model = GPT2LMHeadModel(config)
        
        # 创建ModelArguments
        model_args = ModelArguments(
            use_mt_moslora=True,
            use_mixer=True,
            defuse_gpt2_attn=True,
            ha_lora_r=4,
            ha_lora_alpha=8,
            ha_lora_dropout=0.1,
            hs_lora_r=4,
            hs_lora_alpha=8,
            hs_lora_dropout=0.1,
            hardware_types="v100,xavier,i7",
            target_modules="lm_head"
        )
        
        # 先打印模型结构
        print("Original model structure:")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"  Linear module: {name}")
        
        # 应用MT-MoSLoRA
        model_with_mt_moslora = apply_mt_moslora_to_model(model, model_args)
        
        # 验证模型结构
        mt_moslora_modules = 0
        for name, module in model_with_mt_moslora.named_modules():
            if isinstance(module, MTMoSLoRALinear):
                mt_moslora_modules += 1
                print(f"Found MT-MoSLoRA module: {name}")
        
        assert mt_moslora_modules > 0, "No MT-MoSLoRA modules found"
        
        print(f"✅ Model integration test passed! Found {mt_moslora_modules} MT-MoSLoRA modules")
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_consistency():
    """测试前向传播的一致性"""
    print("\n" + "=" * 60)
    print("Testing Forward Consistency")
    print("=" * 60)
    
    try:
        ha_config = {'r': 4, 'alpha': 8, 'dropout': 0.0, 'use_mixer': True}  # 关闭dropout确保一致性
        hs_config = {'r': 4, 'alpha': 8, 'dropout': 0.0, 'use_mixer': True}
        hardware_types = ['v100', 'xavier']
        
        module = MTMoSLoRALinear(
            in_features=256,
            out_features=256,
            ha_config=ha_config,
            hs_config=hs_config,
            hardware_types=hardware_types
        )
        
        # 测试输入
        x = torch.randn(1, 32, 256)
        
        # 多次前向传播，确保一致性
        outputs = []
        for _ in range(3):
            output = module(x, hardware_id='v100')
            outputs.append(output)
        
        # 检查输出是否一致
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6), f"Output {i} differs from output 0"
        
        print("✅ Forward consistency test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forward consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("MT-MoSLoRA Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("MTMoSLoRALinear Basic Functionality", test_mt_moslora_linear),
        ("Hardware Routing", test_hardware_routing),
        ("Model Integration", test_model_integration),
        ("Forward Consistency", test_forward_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MT-MoSLoRA is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
