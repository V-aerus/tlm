# make_dataset.py 改动记录

## 改动历史

### 版本1：原始版本
- 文件：`make_dataset_backup.py`（已备份）
- 状态：原始工作版本，但存在TVM工作负载注册表在子进程中为空的问题

### 版本2：第一次修复尝试
**改动内容**：
- 在`for_gen`、`for_gen_best`、`for_gen_eval_sketch`三个函数开头添加了简单的重新注册逻辑
- 使用`load_and_register_tasks()`函数重新注册工作负载

**代码示例**：
```python
def for_gen(lines):
    # 在子进程中重新注册工作负载（修复TVM注册表生命周期问题）
    try:
        from common import load_and_register_tasks
        tasks = load_and_register_tasks()
        for task in tasks:
            try:
                auto_scheduler.workload_registry.register_workload_tensors(
                    task.workload_key, task.compute_dag.tensors)
            except:
                pass  # 忽略重复注册错误
    except:
        pass  # 如果注册失败，继续执行
```

**结果**：失败，仍然出现`KeyError: '1bd5a6289c86a9739cb681c2bf0dcd3c'`

### 版本3：第二次修复尝试（当前版本）
**改动内容**：
- 更彻底的修复方法
- 在子进程中重新调用`register_data_path()`来设置全局变量
- 重新加载`all_tasks.pkl`文件并注册工作负载

**代码示例**：
```python
def for_gen(lines):
    # 在子进程中重新注册工作负载（修复TVM注册表生命周期问题）
    try:
        import pickle
        import glob
        import os
        from common import register_data_path
        
        # 从第一行数据中提取target信息
        input, _ = load_record_from_string(lines[0])
        target_str = str(input.task.target)
        
        # 重新注册数据路径
        register_data_path(target_str)
        
        # 重新加载和注册任务
        from common import NETWORK_INFO_FOLDER
        if NETWORK_INFO_FOLDER and os.path.exists(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl"):
            tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))
            for task in tasks:
                try:
                    auto_scheduler.workload_registry.register_workload_tensors(
                        task.workload_key, task.compute_dag.tensors)
                except:
                    pass  # 忽略重复注册错误
    except Exception as e:
        print(f"Warning: Failed to re-register workloads in subprocess: {e}")
        pass  # 如果注册失败，继续执行
```

**结果**：仍然失败，同样的`KeyError`

## 问题分析

### 根本原因
TVM的工作负载注册表（`WORKLOAD_FUNC_REGISTRY`）在子进程中无法正确初始化，这是一个深层的TVM架构问题。

### 可能的解决方案
1. **使用单进程模式**：禁用多进程，但会显著降低处理速度
2. **预处理数据**：在主进程中预处理所有数据，避免在子进程中调用TVM函数
3. **使用不同的数据格式**：避免依赖TVM的`recover_measure_input`函数

## 回滚方案

如果需要回滚到原始版本：
```bash
cp make_dataset_backup.py make_dataset.py
```

## 下一步计划

1. 尝试单进程模式测试
2. 或者采用预处理数据的方法
3. 或者使用之前成功的tokenizer扩展方法













