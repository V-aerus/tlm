import os
import json

utils_json_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/utils.json"  # 使用绝对路径
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

def get_measure_records():
    from common import HARDWARE_PLATFORM
    print(f"Checking utils_json_path: {utils_json_path}")
    if not os.path.exists(utils_json_path):
        print("utils.json does not exist")
        return []
    try:
        with open(utils_json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded utils.json: {data}")
        print(f"HARDWARE_PLATFORM: {HARDWARE_PLATFORM}")
        if HARDWARE_PLATFORM in data and "measure_records" in data[HARDWARE_PLATFORM]:
            print(f"Found measure_records: {data[HARDWARE_PLATFORM]['measure_records']}")
            return data[HARDWARE_PLATFORM]["measure_records"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading utils.json: {e}")
        return []
    print("No measure_records found for HARDWARE_PLATFORM")
    return []