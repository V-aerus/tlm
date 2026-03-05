import os
import json

_DEFAULT_DATA_ROOT = "/home/hehangshuai/workspace/tlm/tlm_dataset/gen"
utils_json_path = os.path.join(os.environ.get("TLM_DATA_ROOT", _DEFAULT_DATA_ROOT), "utils.json")
utils_json = None

def get_utils_json(key):
    from common import HARDWARE_PLATFORM  # 延迟导入
    if not os.path.exists(utils_json_path):
        return []
    global utils_json
    if utils_json is None:
        try:
            with open(utils_json_path, 'r') as f:
                utils_json = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            utils_json = {}
    assert HARDWARE_PLATFORM is not None, "HARDWARE_PLATFORM must be set by calling register_data_path first."
    if HARDWARE_PLATFORM not in utils_json:
        utils_json[HARDWARE_PLATFORM] = {}
    if key and key not in utils_json[HARDWARE_PLATFORM]:
        utils_json[HARDWARE_PLATFORM][key] = []
    return utils_json[HARDWARE_PLATFORM].get(key, [])

def save_utils_json():
    with open(utils_json_path, 'w') as f:
        json.dump(utils_json, f, indent=2)

def add_finetuning_files(file):
    tmp_list = get_utils_json('finetuning_files')
    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()

def get_finetuning_files():
    return get_utils_json('finetuning_files')

def add_test_files(file):
    tmp_list = get_utils_json('test_files')
    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()

def get_test_files():
    return get_utils_json('test_files')

def add_testtuning_files(file):
    tmp_list = get_utils_json('testtuning_files')
    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()

def get_testtuning_files():
    return get_utils_json('testtuning_files')

def get_measure_records(mode: str = "all"):
    from common import HARDWARE_PLATFORM
    debug = os.environ.get("TLM_DEBUG_UTILS", "0") == "1"
    if not os.path.exists(utils_json_path):
        if debug:
            print(f"[utils] utils.json does not exist: {utils_json_path}")
        return []
    try:
        with open(utils_json_path, 'r') as f:
            data = json.load(f)
        if debug:
            print(f"[utils] HARDWARE_PLATFORM: {HARDWARE_PLATFORM}")
        if HARDWARE_PLATFORM in data:
            key = "measure_records"
            if mode == "base":
                key = "measure_records_base"
            elif mode == "kv_lora":
                key = "measure_records_kv_lora"
            if key in data[HARDWARE_PLATFORM]:
                if debug:
                    print(f"[utils] Found {key}: {len(data[HARDWARE_PLATFORM][key])} paths")
                return data[HARDWARE_PLATFORM][key]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if debug:
            print(f"[utils] Error reading utils.json: {e}")
        return []
    if debug:
        print("[utils] No measure_records found for HARDWARE_PLATFORM")
    return []

def add_measure_records(file):
    """添加测量记录文件到utils.json"""
    tmp_list = get_utils_json('measure_records')
    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()
