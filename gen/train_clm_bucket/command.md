
4090，无 bucket，有续训，生成
CUDA_VISIBLE_DEVICES=3 python gen/gen_state.py     --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000     --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json     --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/gen_4090_no_bucket.json     --target 4090     --keep_cnt 16 
***

4090，无 bucket，无续训，生成
CUDA_VISIBLE_DEVICES=3 python gen/gen_state.py     --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_init    --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json     --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_no_bucket.json     --target 4090     --keep_cnt 16 

（v100 v）
CUDA_VISIBLE_DEVICES=3 python gen/gen_state.py     --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_init    --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json     --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_v100_gen_4090_sketch_no_bucket.json     --target nvidia/nvidia-v100     --keep_cnt 16 

4090，有bucket，无续训，生成
CUDA_VISIBLE_DEVICES=3 python gen/gen_state.py     --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_init    --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json     --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_bucket.json     --target 4090     --keep_cnt 16 --use_bucket

（v100 v）
CUDA_VISIBLE_DEVICES=3 python gen/gen_state.py     --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_init    --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json     --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_v100_gen_4090_sketch_bucket.json     --target nvidia/nvidia-v100     --keep_cnt 16 







4090，无 bucket，无续训，测量
CUDA_VISIBLE_DEVICES=3 python gen/measure_programs.py --batch-size=64 --target "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152
-max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32" --to-measure-path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_no_bucket.json --measured-path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_no_bucket_measured_only_bert.json

(v100 v)

CUDA_VISIBLE_DEVICES=0 python measure_programs.py --batch-size=64 --target nvidia/nvidia-v100 --to-measure-path=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/12.9/no_continue_v100_gen_4090_sketch_no_bucket.json --measured-path=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/12.9/no_continue_v100_gen_4090_sketch_no_bucket_measured_only_bert.json

  4090，有 bucket，无续训，测量

  CUDA_VISIBLE_DEVICES=3 python gen/measure_programs.py --batch-size=64 --target "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32" --to-measure-path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_bucket.json  --measured-path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_bucket_measured_only_bert.json

(v100 v)

CUDA_VISIBLE_DEVICES=0 python measure_programs.py --batch-size=64 --target nvidia/nvidia-v100 --to-measure-path=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/12.9/no_continue_v100_gen_4090_sketch_bucket.json  --measured-path=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/12.9/no_continue_v100_gen_4090_sketch_bucket_measured_only_bert.json




4090，无 bucket，无续训，编译

CUDA_VISIBLE_DEVICES=3 TLM_LOG_FILE=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_no_bucket_measured_only_bert.json python gen/tune_relay.py --workload=bert_base --input-shape=\[1,128\] --target "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152
  -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32" --backend=graph

(v100 v)
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/12.9/no_continue_v100_gen_4090_sketch_no_bucket_measured_only_bert.json python tune_relay.py --workload=bert_base --input-shape=\[1,128\] --target=nvidia/nvidia-v100 --backend=graph

4090，有bucket，无续训，编译

CUDA_VISIBLE_DEVICES=3 TLM_LOG_FILE=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/no_continue_gen_4090_bucket_measured_only_bert.json python gen/tune_relay.py --workload=bert_base --input-shape=\[1,128\] --target "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32" --backend=graph

(v100 v)
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/12.9/no_continue_v100_gen_4090_sketch_bucket_measured_only_bert.json python tune_relay.py --workload=bert_base --input-shape=\[1,128\] --target=nvidia/nvidia-v100 --backend=graph