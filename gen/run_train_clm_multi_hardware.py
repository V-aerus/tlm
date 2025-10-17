import subprocess, os

# 设置 screen 命令和相关参数
session_name = os.path.basename(os.path.abspath(__file__))
log_file = f'{session_name}.log'
session_name = session_name.replace('.', '_')

if os.path.exists(log_file):
    tag = input(log_file + ' exist, delete it? [n]')
    if tag == 'y':
        # 删除文件
        os.remove(log_file)

# 构建完整的 screen 命令
cmd = """tmux new -s %s -d '{ 
{ 
set -x
echo "#################################################################"
echo "Starting Multi-Hardware TLM Pre-training"
echo "Hardware: V100, RTX 4090, Xavier, Xeon CPU"
echo "Dataset size: 5.7M programs"
date

export PYTHONUNBUFFERED=1
time CUDA_VISIBLE_DEVICES=0 python train_clm.py \
                                    --do_train \
                                    --model_type=gpt2 \
                                    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_hardware_tokenizer \
                                    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_hardware \
                                    --dataset_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_hardware_pretrain \
                                    --model_name_or_path=gpt2 \
                                    --per_device_train_batch_size=4 \
                                    --overwrite_output_dir=True \
                                    --logging_steps=100 \
                                    --num_train_epochs=1 \
                                    --save_steps=10000 \
                                    --learning_rate=5e-5 \
                                    --warmup_steps=1000 \
                                    --max_steps=100000 \
                                    --gradient_accumulation_steps=4 \
                                    --fp16=True \
                                    --dataloader_num_workers=4

echo "Multi-Hardware Pre-training Completed"
date
} |& tee -a %s 
}' 
""" % (session_name, log_file)

print("Starting multi-hardware TLM pre-training...")
print("Session name:", session_name)
print("Log file:", log_file)
print("Command:", cmd)

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)
