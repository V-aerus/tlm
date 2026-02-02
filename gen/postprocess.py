from dataclasses import dataclass, field
import sys
import re
from transformers import HfArgumentParser
import json
import pickle
import glob
import tqdm
import tvm
from common import register_data_path
import os
from utils import get_measure_records

@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})
    record_mode: str = field(default="all", metadata={"help": "base | kv_lora | all"})
    record_dir: str = field(default=None, metadata={"help": "Optional output dir for measure_records."})
    iter_max: int = field(default=None, metadata={"help": "Optional max iteration index to include (e.g., 1)."})
    iter_list: str = field(default=None, metadata={"help": "Optional comma list of iteration indices to include (e.g., 0,1,2)."})
    clean_output: bool = field(default=False, metadata={"help": "Delete existing output .json files before rebuilding."})
    quiet: bool = field(default=False, metadata={"help": "Reduce output verbosity."})


def _parse_iter_from_path(path: str):
    match = re.search(r"/iter(\d+)", path)
    if not match:
        return None
    return int(match.group(1))


def _filter_by_iter(files, *, iter_max=None, iter_list=None):
    if iter_list:
        allowed = {int(x) for x in iter_list.split(",") if x.strip().isdigit()}
        if not allowed:
            return files
        return [f for f in files if (_parse_iter_from_path(f) in allowed or _parse_iter_from_path(f) is None)]
    if iter_max is None:
        return files
    return [f for f in files if (_parse_iter_from_path(f) is None or _parse_iter_from_path(f) <= iter_max)]


def main():
    parser = HfArgumentParser(ScriptArguments)
    argv = []
    for arg in sys.argv[1:]:
        arg = arg.replace("--record-mode", "--record_mode")
        arg = arg.replace("--record-dir", "--record_dir")
        arg = arg.replace("--iter-max", "--iter_max")
        arg = arg.replace("--iter-list", "--iter_list")
        arg = arg.replace("--clean-output", "--clean_output")
        argv.append(arg)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses(args=argv)[0]
    def log(msg: str):
        if not script_args.quiet:
            print(msg)

    log(script_args)
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)

    from common import MEASURE_RECORD_FOLDER, clean_name
    out_dir = script_args.record_dir or MEASURE_RECORD_FOLDER
    os.makedirs(out_dir, exist_ok=True)
    assert(out_dir is not None)
    if script_args.clean_output:
        old_files = glob.glob(f"{out_dir}/*.json")
        for file in old_files:
            os.remove(file)

    #files = glob.glob(f'{MEASURE_RECORD_FOLDER}/*.json')
    files = []
    measure_records = get_measure_records(script_args.record_mode)
    log(f"Measure records: {measure_records}")
    if measure_records:
        files.extend(measure_records)
    else:
        files = glob.glob(f'{MEASURE_RECORD_FOLDER}/*.json')  # 回退到原有逻辑
    files = _filter_by_iter(files, iter_max=script_args.iter_max, iter_list=script_args.iter_list)
    log(f"Files after measure_records: {files}")  # 添加调试输出
    #for file in files:
        #os.remove(file)
    #files = []
    from utils import get_finetuning_files, get_testtuning_files
    finetuning_files = get_finetuning_files()
    testtuning_files = get_testtuning_files()
    log(f"Finetuning files: {finetuning_files}")  # 添加调试输出
    log(f"Testtuning files: {testtuning_files}")  # 添加调试输出
    files.extend(get_finetuning_files())
    files.extend(get_testtuning_files())
    files = _filter_by_iter(files, iter_max=script_args.iter_max, iter_list=script_args.iter_list)
    log(f"Found files: {files}")

    record_dic = {}
    measured_record_set = set()

    for file in tqdm.tqdm(files, disable=script_args.quiet):
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                json_line = json.loads(line)
                workload_key = json_line["i"][0][0]
                if workload_key not in record_dic:
                    record_dic[workload_key] = []

                i_str = json.dumps(json.loads(line)['i'])
                if i_str in measured_record_set:
                    continue
                else:
                    measured_record_set.add(i_str)
                    record_dic[workload_key].append(line)

    for workload_key, lines in tqdm.tqdm(record_dic.items(), disable=script_args.quiet):
        task_key = (workload_key, str(script_args.target.kind))
        filename = f"{out_dir}/{clean_name(task_key)}.json"
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    from common import HARDWARE_PLATFORM
    log(f"HARDWARE_PLATFORM: {HARDWARE_PLATFORM}")
    assert HARDWARE_PLATFORM is not None
    measured_pkl_path = os.path.join(out_dir, f'measured_{HARDWARE_PLATFORM}.pkl')
    with open(measured_pkl_path, 'wb') as f:
        pickle.dump(measured_record_set, f)


measured_pkl = None
if __name__ == "__main__":
    main()


def check_measured(i_str, measured_pkl_path: str = None):
    global measured_pkl
    if measured_pkl is None:
        from common import HARDWARE_PLATFORM
        assert HARDWARE_PLATFORM is not None
        if measured_pkl_path is None:
            measured_pkl_path = f'measured_{HARDWARE_PLATFORM}.pkl'
        with open(measured_pkl_path, 'rb') as f:
            measured_pkl = pickle.load(f)
    measured = i_str in measured_pkl
    # if measured:
    #     print('measured', end=' ')
    return measured
