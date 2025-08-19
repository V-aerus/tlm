import argparse
import subprocess
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run TLM training with custom parameters.")
parser.add_argument("--model_name_or_path", type=str, default="gen_data/clm_gen_v100", help="Path to pre-trained model or model identifier.")
parser.add_argument("--train_file", type=str, default="gen_data/v100_gen_best/0_merge.json", help="Path to training file.")
parser.add_argument("--per_device_train_batch_size", type=int, default=5, help="Batch size per device.")
parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--output_dir", type=str, default="gen_data/clm_gen_best_v100", help="Output directory for the model.")
parser.add_argument("--tokenizer_name", type=str, default="gen_data/gen_tokenizer_v100", help="Tokenizer name or path.")
parser.add_argument("--model_type", type=str, default="gpt2", help="Model type.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type.")
parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps.")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory.")
parser.add_argument("--remove_unused_columns", action="store_true", help="Remove unused columns.")

# MoSLoRA flags passthrough
parser.add_argument("--use_moslora", action="store_true", help="Enable MoSLoRA via customized PEFT")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument(
    "--target_modules",
    type=str,
    default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj,c_attn,c_fc,c_proj",
    help="Comma-separated module suffixes to adapt",
)
args = parser.parse_args()

# 设置 screen 命令和相关参数
session_name = os.path.basename(os.path.abspath(__file__)).replace('.', '_')
log_file = f'{session_name}.log'

if os.path.exists(log_file):
    os.remove(log_file)

# 构建完整的 train_clm.py 命令
cmd = f"""
tmux new -s {session_name} -d '{{ 
{{ 
set -x
echo "#################################################################"
date

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 python train_clm.py \
    --do_train \
    --model_type={args.model_type} \
    --tokenizer_name={args.tokenizer_name} \
    --output_dir={args.output_dir} \
    --train_file={args.train_file} \
    --per_device_train_batch_size={args.per_device_train_batch_size} \
    --num_train_epochs={args.num_train_epochs} \
    --overwrite_output_dir={str(args.overwrite_output_dir).lower()} \
    --logging_steps={args.logging_steps} \
    --learning_rate={args.learning_rate} \
    --model_name_or_path={args.model_name_or_path} \
    --lr_scheduler_type={args.lr_scheduler_type} \
    --use_moslora={'true' if args.use_moslora else 'false'} \
    --lora_r={args.lora_r} \
    --lora_alpha={args.lora_alpha} \
    --lora_dropout={args.lora_dropout} \
    --target_modules={args.target_modules}

if [ $? -ne 0 ]; then
  curl https://diyi.site/ma\?text\=finish_run_train_clm_best_v100 --noproxy diyi.site
fi

date
}} |& tee -a {log_file} 
}}'
"""

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)