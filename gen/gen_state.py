import glob
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
import tvm
from tvm import auto_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from common import load_and_register_tasks, register_data_path
from hw_kv_aligner import HwKVAligner
from make_dataset import input_to_tokens
from postprocess import check_measured

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Path to base or fully-finetuned model"})
    sketch_path: str = field(metadata={"help": "Path to input sketches (measure records)"})
    save_path: str = field(metadata={"help": "Path to save generated measure records"})
    keep_cnt: int = field(metadata={"help": "Max records to keep per workload"})
    target: str = field(metadata={"help": "Target string or shortcut (e.g., 4090)"})

    tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Optional tokenizer path (defaults to model_path)"})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Optional single LoRA/PEFT adapter path"})
    hardware_embedding_path: str = field(
        default="Embedding/hardware_embeddings_v4_universe.json",
        metadata={"help": "Path to hardware_embeddings_v4_universe.json"},
    )
    target_hardware: Optional[str] = field(
        default=None,
        metadata={"help": "Logical hardware id; if None, inferred from --target"},
    )
    allow_repeat: bool = field(default=True, metadata={"help": "Allow repeated measure inputs"})
    is_build: bool = field(default=False, metadata={"help": "Actually build schedules instead of using placeholders"})
    use_bucket: bool = field(default=False, metadata={"help": "Use bucket tokens when forming text prompts"})
    use_hw_kv: bool = field(default=False, metadata={"help": "Enable HwKVAligner KV injection"})
    hw_kv_aligner_path: Optional[str] = field(default=None, metadata={"help": "Optional HwKVAligner checkpoint path"})


DEFAULT_HW_NAME_MAP = {
    "v100": "nvidia/nvidia-v100",
    "nvidia-v100": "nvidia/nvidia-v100",
    "rtx-4090": "nvidia/rtx-4090",
    "4090": "nvidia/rtx-4090",
    "geforce-rtx-3090": "nvidia/geforce-rtx-3090",
    "3090": "nvidia/geforce-rtx-3090",
    "jetson-agx-xavier": "nvidia/jetson-agx-xavier",
    "xavier": "nvidia/jetson-agx-xavier",
    "xeon": "aws/cpu/c5.18xlarge",
    "c5.18xlarge": "aws/cpu/c5.18xlarge",
}


def load_hardware_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


def resolve_hw_embedding(hw_id: str, embeddings: Dict[str, List[float]]) -> Tuple[Optional[str], Optional[List[float]]]:
    if hw_id in embeddings:
        return hw_id, embeddings[hw_id]

    hw_id_norm = hw_id.lower()
    mapped = DEFAULT_HW_NAME_MAP.get(hw_id_norm)
    if mapped and mapped in embeddings:
        return mapped, embeddings[mapped]

    for name, vec in embeddings.items():
        if hw_id_norm in name.lower():
            return name, vec

    return None, None


def prepare_hw_kv_context(
    use_hw_kv: bool,
    hw_vec: Optional[List[float]],
    model: torch.nn.Module,
    device: torch.device,
    hw_kv_aligner_path: Optional[str] = None,
):
    if not use_hw_kv:
        return None

    if not hw_kv_aligner_path:
        print("[WARN] --use_hw_kv is set but no --hw_kv_aligner_path provided; skip KV injection.")
        return None

    if hw_vec is None:
        print("[WARN] No hardware embedding found; using zeros for HwKVAligner.")
        hw_vec = [0.0] * 24

    hw_vec_t = torch.tensor(hw_vec, dtype=torch.float32, device=device)
    if hw_vec_t.dim() == 1:
        hw_vec_t = hw_vec_t.unsqueeze(0)

    n_layer = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", None))
    n_head = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", None))
    hidden_size = getattr(model.config, "hidden_size", None)
    if n_layer is None or n_head is None or hidden_size is None:
        raise ValueError("Model config missing n_layer/n_head/hidden_size for HwKVAligner.")
    head_dim = hidden_size // n_head

    kv_aligner = HwKVAligner(
        llm_num_layers=n_layer,
        llm_num_heads=n_head,
        llm_head_dim=head_dim,
        hw_dim=hw_vec_t.size(-1),
        num_slots=4,
        backward_depth=min(4, n_layer),
        linker_temperature=1.0,
    )
    if hw_kv_aligner_path:
        try:
            kv_state = torch.load(hw_kv_aligner_path, map_location=device)
            state = kv_state.get("state_dict", kv_state)
            kv_aligner.load_state_dict(state, strict=False)
            print(f"Loaded HwKVAligner checkpoint from {hw_kv_aligner_path}")
        except Exception as e:
            print(f"[WARN] Failed to load HwKVAligner checkpoint: {e}")
    kv_aligner.to(device)
    kv_aligner.eval()
    return {"kv_aligner": kv_aligner, "hw_vec": hw_vec_t}


def load_model_for_inference(args: ScriptArguments):
    print("Loading model for inference with mode detection...")
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter_path")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Model loading completed successfully!")
    return model, tokenizer


def gen_func(task, states, input_item, tokenizer, model, device, gen_kwargs, use_bucket: bool = False, hw_kv_ctx=None):
    if len(states) == 0:
        return []

    tokens = input_to_tokens(
        task,
        states,
        input_item,
        use_bucket=use_bucket,
    )
    tokenizer.padding_side = "left"
    batch = tokenizer(tokens, padding=True, max_length=None)
    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    batch_size = 64

    response_list = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = torch.tensor(input_ids_all[start : start + batch_size], dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_mask_all[start : start + batch_size], dtype=torch.long, device=device)

            # 回归旧版行为：裁掉最后一个 token（通常是结尾 special token），与 gold baseline 保持一致
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

            gen_kwargs["max_new_tokens"] = min(
                gen_kwargs["max_new_tokens"], tokenizer.model_max_length - input_ids.shape[-1]
            )

            extra_kwargs = {}
            input_ids_for_gen = input_ids
            attention_mask_for_gen = attention_mask

            if hw_kv_ctx is not None:
                kv_aligner = hw_kv_ctx["kv_aligner"]
                hw_vec = hw_kv_ctx["hw_vec"].to(device)

                # 1. HW past (force num_beams=1; beam expansion handled by HF generate)
                if hw_vec.size(0) != input_ids.size(0):
                    hw_vec = hw_vec.expand(input_ids.size(0), -1)
                hw_past = kv_aligner(hw_vec, batch_size=input_ids.size(0), num_beams=1)

                # 2. Full mask = HW prefix + prompt
                prefix_len = hw_past[0][0].shape[2]
                prefix_mask = torch.ones(
                    input_ids.size(0),
                    prefix_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

                # 3. Split prefill: keep last token for generate, prefill the rest with HW past
                seq_len = input_ids.size(1)
                if seq_len > 1:
                    input_ids_prefill = input_ids[:, :-1]
                    mask_prefill = full_attention_mask[:, :-1]
                    with torch.no_grad():
                        outputs_prefill = model(
                            input_ids=input_ids_prefill,
                            attention_mask=mask_prefill,
                            past_key_values=hw_past,
                            use_cache=True,
                        )
                    final_past = outputs_prefill.past_key_values
                    input_ids_for_gen = input_ids[:, -1:]  # only last token drives generation
                else:
                    final_past = hw_past
                    input_ids_for_gen = input_ids

                extra_kwargs["past_key_values"] = final_past
                attention_mask_for_gen = full_attention_mask
            else:
                input_ids_for_gen = input_ids
                attention_mask_for_gen = attention_mask

            response = model.generate(
                input_ids=input_ids_for_gen,
                attention_mask=attention_mask_for_gen,
                **gen_kwargs,
                **extra_kwargs,
            )
            gen_start_idx = input_ids_for_gen.shape[-1]
            response = response[:, gen_start_idx:]
            response_list.extend(response.tolist())
    return [tokenizer.batch_decode(item) for item in response_list]


def merge_json_files_safely(tmp_folder, save_path):
    print(f"开始合并JSON文件从 {tmp_folder} 到 {save_path}")

    all_records = []
    part_files = glob.glob(f"{tmp_folder}/*_part")
    part_files.sort()

    total_files = len(part_files)
    valid_files = 0
    total_records = 0

    for part_file in part_files:
        try:
            if os.path.getsize(part_file) > 0:
                valid_files += 1
                file_records = 0
                with open(part_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                                all_records.append(line)
                                file_records += 1
                                total_records += 1
                            except json.JSONDecodeError as e:
                                print(f"警告: {part_file} 第{line_num}行JSON格式错误: {e}")
                                print(f"问题行内容: {line[:100]}...")
                                continue
                print(f"文件 {part_file}: {file_records} 条有效记录")
            else:
                print(f"跳过空文件: {part_file}")
        except Exception as e:
            print(f"处理文件 {part_file} 时出错: {e}")
            continue

    print(f"合并完成: {valid_files}/{total_files} 个文件有效，共 {total_records} 条记录")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(record + "\n")
        print(f"成功写入 {save_path}，共 {len(all_records)} 条记录")
        return len(all_records)
    except Exception as e:
        print(f"写入文件 {save_path} 时出错: {e}")
        raise


def worker(
    err_queue,
    save_path_i,
    sketch_path,
    gen_kwargs,
    model_path,
    tokenizer_path,
    adapter_path,
    target_hardware,
    original_target,
    device,
    allow_repeat,
    keep_cnt,
    is_build,
    worker_id,
    num_workers,
    use_bucket=False,
    hw_kv_cfg=None,
):
    try:
        print(f"Initializing TVM environment in worker for target: {original_target}")
        register_data_path(original_target)
        load_and_register_tasks()

        sketch_path_to_use = sketch_path
        tmp_sanitized = None
        need_sanitize = False
        try:
            with open(sketch_path, "r", encoding="utf-8") as fin:
                first_line = fin.readline()
                if first_line:
                    obj = json.loads(first_line)
                    tgt = obj.get("i", [None])[0][1] if isinstance(obj.get("i"), list) and len(obj["i"]) > 0 else None
                    if isinstance(tgt, str) and " -1 " in tgt:
                        need_sanitize = True
        except Exception:
            pass

        if need_sanitize:
            try:
                import tempfile

                fd, tmp_path = tempfile.mkstemp(prefix=".gen_state_sketch_", suffix=".json")
                os.close(fd)
                with open(sketch_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
                    for line in fin:
                        try:
                            obj = json.loads(line)
                            tgt = obj.get("i", [None])[0][1] if isinstance(obj.get("i"), list) and len(obj["i"]) > 0 else None
                            if isinstance(tgt, str) and " -1 " in tgt:
                                obj["i"][0][1] = tgt.split(" -1 ")[0].strip()
                                line = json.dumps(obj, separators=(",", ":")) + "\n"
                        except Exception:
                            pass
                        fout.write(line)
                sketch_path_to_use = tmp_path
                tmp_sanitized = tmp_path
                print(f"Worker {worker_id}: sanitized target strings for TVM parsing -> {sketch_path_to_use}")
            except Exception as e:
                print(f"Worker {worker_id}: sanitize sketch target failed, fallback to original. Error: {e}")

        script_args = ScriptArguments(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            sketch_path="",
            save_path="",
            keep_cnt=keep_cnt,
            target=original_target,
            adapter_path=adapter_path,
            target_hardware=target_hardware,
            allow_repeat=allow_repeat,
            is_build=is_build,
            use_bucket=use_bucket,
            use_hw_kv=bool(hw_kv_cfg),
            hw_kv_aligner_path=hw_kv_cfg["hw_kv_aligner_path"] if hw_kv_cfg else None,
            hardware_embedding_path="",
        )

        print(f"Loading model and tokenizer in worker process for device: {device}")
        model, tokenizer = load_model_for_inference(script_args)

        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        print(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()

        hw_kv_ctx = None
        if hw_kv_cfg:
            hw_kv_ctx = prepare_hw_kv_context(
                use_hw_kv=True,
                hw_vec=hw_kv_cfg.get("hw_vec"),
                model=model,
                device=device,
                hw_kv_aligner_path=hw_kv_cfg.get("hw_kv_aligner_path"),
            )

        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        if os.path.exists(save_path_i):
            os.remove(save_path_i)

        print(f"Worker {worker_id}: Reading and processing sketches from {sketch_path_to_use}")
        inputs, _ = auto_scheduler.RecordReader(sketch_path_to_use).read_lines()
        sketch_dic = {}
        inp_dic = {}
        for inp in tqdm.tqdm(inputs):
            workload_key = inp.task.workload_key
            inp_str = inp.to_json()
            if inp_str in inp_dic:
                inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
            else:
                inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
                inp_dic[inp_str] = inp
            if workload_key not in sketch_dic:
                sketch_dic[workload_key] = []
            sketch_dic[workload_key].append(inp)

        sketch_dic_list_full = list(sketch_dic.items())
        per_len = math.ceil(len(sketch_dic_list_full) / num_workers)
        start_idx = worker_id * per_len
        end_idx = min((worker_id + 1) * per_len, len(sketch_dic_list_full))
        my_sketch_chunk = sketch_dic_list_full[start_idx:end_idx]

        print(f"Worker {worker_id}: 数据分片 [{start_idx}:{end_idx}]，处理 {len(my_sketch_chunk)} out of {len(sketch_dic_list_full)} workload groups")
        total_workloads = len(my_sketch_chunk)
        successful_workloads = 0
        total_generated = 0
        print(f"Worker {worker_id}: 开始处理 {total_workloads} 个workload组")

        for workload_idx, (workload_key, inputs_for_key) in enumerate(tqdm.tqdm(my_sketch_chunk, desc=f"Worker {worker_id}")):
            try:

                def gen_func_inner(task, states, max_new_tokens):
                    max_new_tokens = max(max_new_tokens, 1)
                    gen_kwargs["max_new_tokens"] = max_new_tokens
                    return gen_func(
                        task,
                        states,
                        inputs_for_key[0],
                        tokenizer,
                        model,
                        device,
                        gen_kwargs,
                        use_bucket=use_bucket,
                        hw_kv_ctx=hw_kv_ctx,
                    )

                policy = auto_scheduler.SketchPolicy(inputs_for_key[0].task)
                measure_inputs = []
                measure_results = []
                input_set = set()

                retry_i = 0
                while retry_i < 5:
                    try:
                        all_state_list = policy.gen_states([inp.state for inp in inputs_for_key], gen_func_inner)

                        measure_inputs_tmp = []
                        for state in all_state_list:
                            inp = auto_scheduler.MeasureInput(inputs_for_key[0].task, state)
                            i_str = inp.to_json()
                            if i_str in input_set:
                                continue
                            if allow_repeat is False and check_measured(i_str):
                                continue

                            input_set.add(i_str)
                            measure_inputs_tmp.append(inp)

                        default_build_result = auto_scheduler.measure.BuildResult(None, [], 0, None, 0)
                        if is_build:
                            build_results = builder.build(measure_inputs_tmp)
                        else:
                            build_results = [default_build_result for _ in measure_inputs_tmp]
                        for res, inp in zip(build_results, measure_inputs_tmp):
                            if res.error_no == 0:
                                measure_inputs.append(inp)
                                measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

                        retry_i += 1
                        if len(measure_inputs) >= keep_cnt:
                            break
                    except Exception as e:
                        print(f"Worker {worker_id}: workload {workload_key} 第{retry_i+1}次重试时出错: {e}")
                        retry_i += 1
                        if retry_i >= 5:
                            print(f"Worker {worker_id}: workload {workload_key} 重试5次后仍然失败，跳过")
                            break
                        continue

                if len(measure_inputs) > keep_cnt:
                    measure_inputs, measure_results = zip(*random.sample(list(zip(measure_inputs, measure_results)), keep_cnt))

                if len(measure_inputs) > 0:
                    auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)
                    successful_workloads += 1
                    total_generated += len(measure_inputs)
                    print(f"Worker {worker_id}: workload {workload_key} 成功生成 {len(measure_inputs)} 条记录")
                else:
                    print(f"Worker {worker_id}: workload {workload_key} 未能生成任何有效记录")

            except Exception as e:
                print(f"Worker {worker_id}: 处理workload {workload_key} 时发生严重错误: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"Worker {worker_id}: 处理完成！成功处理 {successful_workloads}/{total_workloads} 个workload，共生成 {total_generated} 条记录")
        if tmp_sanitized:
            try:
                os.remove(tmp_sanitized)
            except Exception:
                pass
    except Exception as e:
        err_queue.put(e)


def main():
    import tempfile
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    gen_kwargs = {
        "min_length": -1,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": None,
        "eos_token_id": None,
        "max_new_tokens": 512,
    }

    hw_kv_cfg = None
    if script_args.use_hw_kv:
        embeddings = load_hardware_embeddings(script_args.hardware_embedding_path)
        hw_id_candidate = script_args.target_hardware or str(script_args.target)
        hw_name, hw_vec = resolve_hw_embedding(hw_id_candidate, embeddings)
        if hw_vec is None:
            print(f"[WARN] No embedding found for hw_id={hw_id_candidate}; using zeros.")
        hw_kv_cfg = {
            "use_hw_kv": True,
            "hw_vec": hw_vec,
            "hw_kv_aligner_path": script_args.hw_kv_aligner_path,
        }

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        visible_gpu_list = [int(x.strip()) for x in visible_devices.split(",") if x.strip()]
        num_gpus = len(visible_gpu_list)
        print(f"使用CUDA_VISIBLE_DEVICES={visible_devices}，检测到{num_gpus}个GPU")
        print(f"可见GPU列表: {visible_gpu_list}")
    else:
        num_gpus = torch.cuda.device_count()
        visible_gpu_list = list(range(num_gpus))
        print(f"未设置CUDA_VISIBLE_DEVICES，使用所有{num_gpus}个GPU")

    processes = []
    tmp_folder = tempfile.mkdtemp(prefix=".gen_state_")
    err_queue = Queue()
    for gpu_i in range(num_gpus):
        save_path_i = f"{tmp_folder}/{gpu_i}_part"
        device = f"cuda:{gpu_i}"
        if visible_devices:
            actual_physical_gpu = visible_gpu_list[gpu_i]
            print(f"Worker {gpu_i}: 使用设备 {device} (物理GPU {actual_physical_gpu})")
        else:
            print(f"Worker {gpu_i}: 使用设备 {device}")
        p = Process(
            target=worker,
            args=(
                err_queue,
                save_path_i,
                script_args.sketch_path,
                gen_kwargs,
                script_args.model_path,
                script_args.tokenizer_path,
                script_args.adapter_path,
                script_args.target_hardware,
                script_args.target,
                device,
                script_args.allow_repeat,
                script_args.keep_cnt,
                script_args.is_build,
                gpu_i,
                num_gpus,
                script_args.use_bucket,
                hw_kv_cfg,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if not err_queue.empty():
        raise Exception(f"An exception occurred in the child process: {err_queue.get()}")

    try:
        total_records = merge_json_files_safely(tmp_folder, script_args.save_path)
        print(f"推理完成！总共生成 {total_records} 条记录")
    except Exception as e:
        print(f"JSON合并失败: {e}")
        print("回退到原始cat命令...")
        subprocess.run(f"cat {tmp_folder}/*_part > {script_args.save_path}", shell=True)
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
