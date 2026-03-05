import pickle
from tvm import auto_scheduler
import re
import glob
import os

NETWORK_INFO_FOLDER = None
TO_MEASURE_PROGRAM_FOLDER = None
MEASURE_RECORD_FOLDER = None
HARDWARE_PLATFORM = None


def _resolve_data_root():
    """Resolve dataset root in a portable way.

    Priority:
      1) DATA_ROOT env (expected: .../tlm_dataset/gen)
      2) TLM_ROOT env + /tlm_dataset/gen
      3) infer from current file path (repo_root/tlm_dataset/gen)
    """
    data_root = os.environ.get("DATA_ROOT", "").strip()
    if data_root:
        return data_root

    tlm_root = os.environ.get("TLM_ROOT", "").strip()
    if tlm_root:
        return os.path.join(tlm_root, "tlm_dataset", "gen")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "tlm_dataset", "gen")

def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace("\"", "")
    x = x.replace("'", "")
    return x

def register_data_path(target_str):
    assert(isinstance(target_str, str))
    model_list = ['i7', 'v100', 'a100', '2080', '4090', '3090', 'xavier', 'orin', 'ryzen5800h', 'xeon', 'multi', 'None']
    alias_map = {
        # 3090 直接复用 4090 的数据目录/网络信息
        '3090': '4090',
        'nvidia/nvidia-3090': '4090',
    }
    model = 'None'
    for m in model_list:
        if m.lower() in target_str.lower():  # 忽略大小写，提高兼容性
            model = m
            break
    if model == 'None':
        # 兼容未显式带型号的 canonical target 串，依据 arch/mcpu 进行推断
        ts = target_str.lower()
        if "arch=sm_86" in ts:
            model = "4090"
        elif "arch=sm_80" in ts or "a100" in ts:
            model = "a100"
        elif "arch=sm_70" in ts:
            model = "v100"
        elif "arch=sm_75" in ts:
            model = "2080"
        elif "arch=sm_87" in ts or "orin" in ts:
            model = "orin"
        elif "arch=sm_72" in ts or "carmel" in ts:
            model = "xavier"
        elif "mcpu=znver3" in ts or "mcpu=znver2" in ts or "mcpu=znver1" in ts or "5800h" in ts or "ryzen" in ts:
            model = "ryzen5800h"
        elif "skylake-avx512" in ts or "mcpu=skylake" in ts:
            model = "xeon"
    # 将别名映射到实际目录
    model = alias_map.get(model, model)
    assert(model != 'None')

    print(f'register data path: {model}')
    global NETWORK_INFO_FOLDER, TO_MEASURE_PROGRAM_FOLDER, MEASURE_RECORD_FOLDER, HARDWARE_PLATFORM
    data_root = _resolve_data_root()
    NETWORK_INFO_FOLDER = f"{data_root}/dataset/network_info/{model}"
    TO_MEASURE_PROGRAM_FOLDER = f"{data_root}/dataset/to_measure_programs/{model}"
    MEASURE_RECORD_FOLDER = f"{data_root}/dataset/measure_records/{model}"
    HARDWARE_PLATFORM = model


def get_relay_ir_filename(target, network_key):
    assert(NETWORK_INFO_FOLDER is not None)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"


def get_task_info_filename(network_key, target):
    assert(NETWORK_INFO_FOLDER is not None)
    network_task_key = (network_key,) + (str(target.kind),)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"


def load_tasks_path(target):
    assert(NETWORK_INFO_FOLDER is not None)
    files = glob.glob(f"{NETWORK_INFO_FOLDER}/*{target.kind}*.pkl")
    return files


def load_and_register_tasks():
    assert(NETWORK_INFO_FOLDER is not None)
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


def get_to_measure_filename(task):
    assert(TO_MEASURE_PROGRAM_FOLDER is not None)
    task_key = (task.workload_key, str(task.target.kind))
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


def get_measure_record_filename(task, target=None):
    assert(MEASURE_RECORD_FOLDER is not None)
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{MEASURE_RECORD_FOLDER}/{clean_name(task_key)}.json"


def hold_out_task_files(target, only_bert=False):
    if only_bert:
        files = {
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target)
        }
    else:
        files = {
            "resnet_50": get_task_info_filename(('resnet_50', [1,3,224,224]), target),
            "mobilenet_v2": get_task_info_filename(('mobilenet_v2', [1,3,224,224]), target),
            "resnext_50": get_task_info_filename(('resnext_50', [1,3,224,224]), target),
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target),
            # "gpt2": get_task_info_filename(('gpt2', [1,128]), target),
            # "llama": get_task_info_filename(('llama', [4,256]), target),
            "bert_tiny": get_task_info_filename(('bert_tiny', [1,128]), target),
            
            "densenet_121": get_task_info_filename(('densenet_121', [8,3,256,256]), target),
            "bert_large": get_task_info_filename(('bert_large', [4,256]), target),
            "wide_resnet_50": get_task_info_filename(('wide_resnet_50', [8,3,256,256]), target),
            "resnet3d_18": get_task_info_filename(('resnet3d_18', [4,3,144,144,16]), target),
            "dcgan": get_task_info_filename(('dcgan', [8,3,64,64]), target)
        }
    return files


def yield_hold_out_five_files(target, only_bert=False):
    files = hold_out_task_files(target, only_bert=only_bert)

    for workload, file in files.items():
        tasks_part, task_weights = pickle.load(open(file, "rb"))
        for task, weight in zip(tasks_part, task_weights):
            yield workload, task, get_measure_record_filename(task, target), weight


def get_hold_out_five_files(target):
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target))]))
    files.sort()
    return files


def get_bert_files(target):
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target, True))]))
    files.sort()
    return files


def get_ansor_eval_files(target):
    """Return measure record filenames for Ansor baseline networks.

    Default set: bert_base / resnet_50 / mobilenet_v2 / inception_v3(1x3x299x299).
    """
    selected = []
    for workload, _, record_file, _ in yield_hold_out_five_files(target, only_bert=False):
        if workload in ("bert_base", "resnet_50", "mobilenet_v2"):
            selected.append(record_file)
    # Inception v3 is not part of hold_out_task_files(), so append from network_info directly.
    inception_task_file = get_task_info_filename(("inception_v3", [1, 3, 299, 299]), target)
    if os.path.exists(inception_task_file):
        tasks_part, task_weights = pickle.load(open(inception_task_file, "rb"))
        for task, _weight in zip(tasks_part, task_weights):
            selected.append(get_measure_record_filename(task, target))
    selected = list(set(selected))
    selected.sort()
    return selected
