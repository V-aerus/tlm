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

#--resume_from_checkpoint=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_base_xavier_continued/checkpoint-108000 \
# 构建完整的 screen 命令
cmd = """tmux new -s %s -d '{ 
{ 
set -x
echo "#################################################################"
date

export PYTHONUNBUFFERED=1
time CUDA_VISIBLE_DEVICES=0 python train_clm.py \
                                    --do_train \
                                    --model_type=gpt2 \
                                    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
                                    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_bast_v100_test_v1 \
                                    --dataset_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v1 \
                                    --model_name_or_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_v100 \
                                    --per_device_train_batch_size=5 \
                                    --overwrite_output_dir=False \
                                    --logging_steps=100 \
                                    --num_train_epochs=1 \
                                    --save_steps=8000


date
} |& tee -a %s 
}' 
""" % (session_name, log_file)
# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)