#!/usr/bin/env python3
"""
多硬件TLM预训练脚本
基于run_train_clm_origin.py，适配多硬件tokenizer和数据集
"""
import subprocess, os

# 设置 tmux 会话和日志
session_name = os.path.basename(os.path.abspath(__file__))
log_file = f'{session_name}.log'
session_name = session_name.replace('.', '_')

if os.path.exists(log_file):
    tag = input(log_file + ' exist, delete it? [n]')
    if tag == 'y':
        # 删除文件
        os.remove(log_file)

# 构建完整的 tmux 命令
cmd = """tmux new -s %s -d '{ 
{ 
set -x
echo "#################################################################"
echo "开始多硬件TLM预训练"
echo "Tokenizer: 9,447 tokens (V100 + Xavier + RTX4090 + Xeon)"
echo "Dataset: 5,610,292 samples"
echo "GPU: CUDA:1"
date

export PYTHONUNBUFFERED=1
time CUDA_VISIBLE_DEVICES=1 python train_clm.py \\
                                    --do_train \\
                                    --model_type=gpt2 \\
                                    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \\
                                    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \\
                                    --dataset_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/pretrain_data_multi_v1 \\
                                    --per_device_train_batch_size=4 \\
                                    --overwrite_output_dir=True \\
                                    --logging_steps=100 \\
                                    --num_train_epochs=3 \\
                                    --remove_unused_columns=False \\
                                    --learning_rate=5e-5 \\
                                    --save_steps=8000

echo "#################################################################"
echo "多硬件TLM预训练完成"
date
} |& tee -a %s 
}' 
""" % (session_name, log_file)

print("🚀 启动多硬件TLM预训练")
print(f"📊 配置信息:")
print(f"   • Tokenizer: 9,447 tokens")
print(f"   • 数据集: 560万+ 样本")
print(f"   • GPU: CUDA:1")
print(f"   • 批次大小: 4")
print(f"   • 训练轮数: 3")
print(f"   • 学习率: 5e-5")
print(f"📝 日志文件: {log_file}")
print(f"🖥️  tmux会话: {session_name}")
print()
print("启动中...")

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)

print("✅ 预训练已在tmux会话中启动！")
print(f"💡 查看进度: tmux attach-session -t {session_name}")
print(f"💡 查看日志: tail -f {log_file}")











