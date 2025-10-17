#!/bin/bash

# MT-MoSLoRA 完整迭代训练流程
# 从草图生成到v2版本训练的完整流程

echo "🚀 MT-MoSLoRA 迭代训练完整流程启动"
echo "=================================="

# 设置错误处理
set -e

echo "📋 迭代训练步骤："
echo "1. 生成新的训练草图"
echo "2. 使用MT-MoSLoRA v1生成张量程序"
echo "3. 测量生成程序的性能"
echo "4. 整理数据并构建新的SFT数据集"
echo "5. 训练MT-MoSLoRA v2"
echo ""

read -p "是否开始执行完整迭代流程？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 用户取消执行"
    exit 1
fi

echo ""
echo "🔄 开始执行迭代训练流程..."

# 步骤1: 生成草图
echo ""
echo "=== 步骤1: 生成新的训练草图 ==="
chmod +x run_iterative_gen_sketches.sh
./run_iterative_gen_sketches.sh

# 步骤2: 生成张量程序
echo ""
echo "=== 步骤2: 生成张量程序 ==="
chmod +x run_iterative_gen_programs.sh
./run_iterative_gen_programs.sh

# 步骤3: 性能测量
echo ""
echo "=== 步骤3: 性能测量 ==="
chmod +x run_iterative_measure.sh
./run_iterative_measure.sh

# 步骤4: 数据整理
echo ""
echo "=== 步骤4: 数据整理和SFT数据集构建 ==="
chmod +x run_iterative_postprocess.sh
./run_iterative_postprocess.sh

# 步骤5: 迭代训练
echo ""
echo "=== 步骤5: 训练MT-MoSLoRA v2 ==="
chmod +x run_mt_moslora_iterative_v2.sh
./run_mt_moslora_iterative_v2.sh

echo ""
echo "🎉 MT-MoSLoRA 迭代训练完成！"
echo "=================================="
echo "新模型版本: v2_mt_moslora_grouped"
echo "位置: /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_2_mt_moslora_grouped"
echo ""
echo "📊 迭代效果对比："
echo "- 使用MT-MoSLoRA v1生成的程序质量应该比基础模型更好"
echo "- v2版本应该比v1版本有进一步的性能提升"
echo "- 可以通过性能测量数据验证迭代效果"
