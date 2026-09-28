# Linux GPU 服务器部署与扩训完整使用手册

> 2026-09-23 新增：全模型训练、实时日志、GPU 队列与自动对比请优先使用[服务器全模型训练与性能对比操作指南](服务器全模型训练与性能对比操作指南.md)。本篇保留旧单模型入口的迁移流程；旧 `train.py` 的恢复限制不代表新入口已经实现精确续训。

更新日期：2026-09-20。项目：CPN-HRL-DAG（optest）。

> 本文基于当前仓库实际代码编写，覆盖迁移、安装、数据核验、配置生成、GPU 训练、完整验证、可视化、扩容和故障排查。本文档不会自动执行其中命令；新配置需要你在服务器运行生成命令后才存在。本文没有修改训练算法，也没有实现文末列出的待改造功能。
>
> 固定要求：始终只在原 validation 全量 **108 场景／54 个基础 DAG** 上验证；不使用 proxy，不运行 test，不重新随机切分验证集。示例环境为 Bash、Python 3.12、NVIDIA GPU；服务器型号、驱动和调度系统尚未核实。

## 目录

1. 当前状态与推荐路线
2. 上传文件与迁移校验
3. Linux 环境安装
4. 数据与固定协议预检
5. 配置生成与迁移验收
6. GPU 训练与作业管理
7. 监控、中断与恢复
8. 完整评估与可视化
9. 扩大训练数据
10. 多种子与图学习基线
11. 六方法比较的边界
12. 常见问题及解决办法
13. 归档与最终验收
14. 建议优先实施的工程改造

## 1. 当前状态与推荐路线

### 1.1 历史参照

历史六方法评估于 2026-09-18 完成，2026-09-20 完成报告整理。该轮复用了已有 checkpoint，没有重新训练；每种方法均在 108 个场景上得到有效调度。

| 方法 | 平均 makespan / HEFT makespan |
| --- | ---: |
| HEFT | 1.000000 |
| Graph PPO | 1.016401 |
| Tier MAPPO | 1.098707 |
| Residual HRL（主模型） | 0.995168 |
| HRL-safe（主模型加安全候选选择） | 0.953510 |
| beam + blocks（独立搜索，不含神经候选） | 0.918681 |

主模型历史预算为 150 episodes，最佳 episode=100；两种图学习基线各为 2592 episodes，最佳 epoch=2。不是等预算架构消融。详细结果见 [历史统一比较报告](主模型与基线统一对比结果_2026-09-20.md)。

### 1.2 推荐执行顺序

| 阶段 | 操作 | 通过条件 |
| --- | --- | --- |
| A | 上传代码、数据、manifest、历史产物 | 文件完整，关键 hash 一致 |
| B | 安装 Python 环境与 CUDA 版 PyTorch | CUDA 前后向测试和项目测试通过 |
| C | 历史 checkpoint 复核、GPU 短跑 | 全量 108 验证有效，无 GPU 错误 |
| D | 原规模 50/100/300 扩至 3000 episodes | 训练预算、日志、最终报告完整 |
| E | 独立 CPU 验证与可视化 | 身份、HEFT 分母和有效率一致 |
| F | 新增 500，再试 1000 任务训练 | 原验证不变，无泄漏，资源足够 |
| G | 多种子与公平预算比较 | 先使比较入口支持目标协议 |

先完成 A–E 就能完整运行当前主模型。第一次不要同时扩大数据、模型容量、算法和多卡并行。

### 1.3 已支持与未支持

已支持：主模型 CUDA 训练、课程筛选、全量 validation、基础 checkpoint 恢复、独立评估、绘图。

尚未完整支持：精确断点重放、仅加载权重的新微调入口、无放回全覆盖采样、按规模指定采样比例、多卡 DDP、任意扩容数据集的六方法比较。

主模型入口只提供 `--config` 与 `--resume`；不要使用不存在的 `--device`、`--epochs` 或 `--init-from`。主模型设备与预算在 YAML 设置。

## 2. 上传文件与迁移校验

### 2.1 服务器预检

```bash
uname -a
cat /etc/os-release
nvidia-smi
lscpu
free -h
df -h .
command -v python3.12 || command -v python3
command -v sbatch || true
```

记录 GPU 型号、显存、驱动、分配到的 CPU、内存、磁盘和是否有 Slurm。集群必须申请计算节点，不在登录节点训练，不擅自升级共享服务器驱动。

当前代码会加载场景并缓存 episode 轨迹，大图可能同时增大主机内存、显存和模拟耗时。不能只看模型参数估计资源，也不能在未知硬件下保证训练完成时间。

### 2.2 上传范围

必需：`src/`、`scripts/`、`configs/`、`tests/`、`pyproject.toml`、`data/raw/zenodo-18927122-derived/`、`data/manifests/`。

建议保留：`README.md`、`doc/`、`docs/`，以及以下历史目录：

- `outputs/zenodo_heft_safe_fast_2026/`：主模型、配置、日志。
- `outputs/graph_baselines_cpu_20260917/`：两种基线模型与配置。
- `outputs/main_model_comparison_20260918/`：历史比较参照。

不用上传 Windows 环境、`.idea/`、`__pycache__/`、`.pytest_cache/`。原始 `.tar.xz` 无需解压，适配器支持读取压缩包。

可通过已有 SFTP 工具上传，也可在 Windows PowerShell 使用 OpenSSH。先把用户名和地址替换为真实值：

```powershell
Set-Location D:\programing\python\optest
$Remote = 'your_user@your_server'
ssh $Remote 'mkdir -p ~/optest/outputs'
scp -r src scripts configs tests data doc docs pyproject.toml README.md "${Remote}:~/optest/"
scp -r outputs/zenodo_heft_safe_fast_2026 outputs/graph_baselines_cpu_20260917 outputs/main_model_comparison_20260918 "${Remote}:~/optest/outputs/"
```

非默认 SSH 端口：`ssh -p PORT`，`scp -P PORT`。上传中断后核验并补传，不将部分文件当成完整数据。

### 2.3 关键文件 hash

本机：

```powershell
Get-FileHash data/manifests/grapheonrl_mixed_iid_seed7.json -Algorithm SHA256
Get-FileHash outputs/zenodo_heft_safe_fast_2026/best.pt -Algorithm SHA256
```

服务器：

```bash
cd ~/optest
sha256sum data/manifests/grapheonrl_mixed_iid_seed7.json
sha256sum outputs/zenodo_heft_safe_fast_2026/best.pt
```

当前已记录：

```text
manifest:
421315d01e2c7336a3f812e1ce8b39a491f4a7e5390519dd33fe2ca90c955bca
main best.pt:
9c39698ae0f56ab82e6b62e12897182960259027846bd010014db9e976559d9b
```

若迁移前主动更新了文件，以实际源文件 hash 为准并记录原因。不要为绕过比较检查改写 manifest 或 checkpoint 元数据。

## 3. Linux 环境安装

### 3.1 创建环境

以下 Bash 命令都在项目根目录执行。每次新会话重新激活环境。

```bash
cd ~/optest
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs environment
```

若没有 Python 3.12/venv，使用管理员提供的模块或 Conda 环境，不修改系统默认 Python。Python 3.12 是迁移建议，不代表任何 PyTorch 版本都适配它。

### 3.2 安装 GPU PyTorch

先根据 GPU 架构和驱动，在官方安装选择器选择 Linux、Pip、对应 CUDA 构建：

```text
https://pytorch.org/get-started/locally/
```

执行它给出的安装命令，然后安装项目。这里不硬编码 CUDA wheel 地址，因为服务器驱动和 GPU 未知；不要直接照搬 Windows Torch 构建。

```bash
python -m pip install -e '.[train,analysis,dev]'
python -m pip check
python - <<'PY'
import sys
import torch
print('Python:', sys.version)
print('Torch:', torch.__version__)
print('Torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Stop: CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
probe = torch.ones((64, 64), device='cuda', requires_grad=True)
probe.square().mean().backward()
torch.cuda.synchronize()
print('CUDA forward/backward: OK')
PY
python -m pytest -q
```

本地当前版本曾通过 125 项测试，服务器结果以本次输出为准。小张量 CUDA 检查不能替代真实 LSTM/PPO 短跑。`nvidia-smi` 的 CUDA 信息不能单独当作 Python 中实际加载的 runtime 版本。

### 3.3 记录环境

```bash
python -m pip freeze > environment/pip-freeze-linux.txt
python -m torch.utils.collect_env > environment/torch-env.txt
nvidia-smi > environment/nvidia-smi.txt
uname -a > environment/uname.txt
```

不要将 Windows freeze 文件直接用于 Linux 重建。分享环境日志前检查是否包含不宜公开的主机信息。

## 4. 数据与固定协议预检

### 4.1 原始数据完整性

```bash
ls data/raw/zenodo-18927122-derived/
head -n 5 data/raw/zenodo-18927122-derived/checksums.sha256
(
  cd data/raw/zenodo-18927122-derived
  sha256sum -c checksums.sha256
)
```

若清单包含未上传的可选包，补齐或明确登记缺失，不忽略所有校验错误。当前本地清单包含 stg_to_json.tar.xz，但数据目录未包含该转换工具包；这不等于训练所用 rnc 数据包损坏。应区分可选工具包缺失与所选规模／system_configs 数据缺失。若路径布局不同，先检查清单和工作目录，不修改数据内容“修复”路径问题。

### 4.2 固定划分核验

下面只加载数据和检查身份，不训练，也不运行 test 评估。可能花费一定时间和内存。

```bash
python - <<'PY'
from pathlib import Path
from cpn_hrl_dag.utils.config import load_config
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits
config = load_config('configs/zenodo_graph_ppo_comparison.yaml')
assert Path(config['dataset']['split_manifest']).is_file()
splits, manifest = load_fixed_splits(config)
for name, scenarios in splits.items():
    print(name, len(scenarios), 'scenarios',
          len({item.metadata['original_graph_id'] for item in scenarios}), 'base DAGs')
assert len(splits['validation']) == 108
print('Fixed protocol OK; no test evaluation')
PY
```

| 划分 | 场景 | 基础 DAG |
| --- | ---: | ---: |
| train | 864 | 432 |
| validation | 108 | 54 |
| test | 108 | 54 |

读取 test 身份检查泄漏不等于评估 test。所有后续评估命令显式指定 `--split validation`。

**主模型入口在 manifest 不存在时会自动生成划分。必须提前检查固定文件存在，不能让自动创建替代原协议。**

## 5. 配置生成与迁移验收

### 5.1 生成短跑和 3000-episode 配置

以下脚本深复制已有主模型配置，保留模型与 PPO 参数，调整预算和验证设置；文件存在时拒绝覆盖。

```bash
python - <<'PY'
from copy import deepcopy
from pathlib import Path
import yaml
base = yaml.safe_load(Path('configs/zenodo_heft_safe_fast_2026.yaml').read_text(encoding='utf-8'))
for name, high_episodes, joint_episodes, interval in [
    ('main_gpu_smoke_seed2026', 2, 2, 2),
    ('main_gpu_3000_seed2026', 600, 2400, 100),
]:
    config = deepcopy(base)
    config['experiment'] = {'name': name, 'seed': 2026}
    config['dataset'].pop('limit_per_dataset', None)
    config['dataset']['archive_task_counts'] = [50, 100, 300]
    config['dataset']['split_manifest'] = 'data/manifests/grapheonrl_mixed_iid_seed7.json'
    config['device'] = 'cuda'
    config['device_options'] = {'disable_cudnn': True}
    config['training'].update(low_pretrain_episodes=0, high_train_episodes=high_episodes,
        joint_train_episodes=joint_episodes, evaluation_interval=interval, early_stop_patience=30)
    config['evaluation'].update(evaluate_initial=True, deterministic=True, proxy_scenarios=0)
    config['curriculum']['stages'] = [
        {'min_tasks': 50, 'max_tasks': 100, 'episodes': high_episodes},
        {'min_tasks': 50, 'max_tasks': 300, 'episodes': joint_episodes},
    ]
    config['output_dir'] = f'outputs/{name}'
    assert not Path(config['output_dir']).exists(), config['output_dir']
    assert Path(config['dataset']['split_manifest']).is_file()
    with Path(f'configs/{name}.yaml').open('x', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(name)
PY
```

说明：

- `proxy_scenarios: 0` 表示完整验证，而不是关闭验证。
- smoke 仅减少训练 episodes，每次验证仍是 108 场景。
- 600 高层＋2400 联合是建议起始预算，不是已验证最优参数。
- `early_stop_patience: 30` 避免沿用原值 4 导致扩训过早停止。
- 初始和最终选中模型也会全量验证；短跑不一定很快完成。
- 暂时禁用 cuDNN 是针对本项目历史 CUDA 故障的保守选择，不代表已证明 Linux 必须如此。
- 当前采样有放回，课程只过滤规模范围，不保证遍历全部场景或规模均衡。

### 5.2 建议先复核旧 checkpoint

生成独立 CPU 配置，防止评估入口覆盖历史训练目录中的配置：

```bash
python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('outputs/zenodo_heft_safe_fast_2026/config.yaml').read_text(encoding='utf-8'))
config['device'] = 'cpu'
config['dataset']['roots']['grapheonrl'] = 'data/raw/zenodo-18927122-derived'
config['dataset']['grapheonrl_system_configs'] = 'data/raw/zenodo-18927122-derived'
config['dataset']['split_manifest'] = 'data/manifests/grapheonrl_mixed_iid_seed7.json'
config['evaluation']['proxy_scenarios'] = 0
config['output_dir'] = 'outputs/migration_reference_cpu'
assert not Path(config['output_dir']).exists()
with Path('configs/migration_reference_cpu.yaml').open('x', encoding='utf-8') as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/evaluate.py \
  --config configs/migration_reference_cpu.yaml --policy hrl \
  --checkpoint outputs/zenodo_heft_safe_fast_2026/best.pt --split validation
```

检查 `outputs/migration_reference_cpu/eval_hrl_validation/summary.json`，均值应接近历史 0.995168。明显差异先核查 scenario ID、HEFT 分母、配置和模型 hash；跨平台微小数值变化需记录，不通过改验证集对齐成绩。

## 6. GPU 训练与作业管理

### 6.1 前台冒烟

```bash
set -o pipefail
test ! -e outputs/main_gpu_smoke_seed2026 || { echo 'Output exists; stop'; exit 1; }
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u scripts/train.py --config configs/main_gpu_smoke_seed2026.yaml \
  2>&1 | tee logs/main_gpu_smoke_seed2026.log
```

线程 4 只是起始值，不超过作业分配 CPU。通过条件：退出码 0；checkpoint、训练日志与报告齐全；108 场景／54 DAG；有效调度率 1.0；无 CUDA、NaN、OOM 错误。失败目录保留，下次用新运行名，不覆盖证据。

### 6.2 普通独占服务器正式训练

```bash
test ! -e outputs/main_gpu_3000_seed2026 || { echo 'Output exists; stop'; exit 1; }
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  nohup python -u scripts/train.py --config configs/main_gpu_3000_seed2026.yaml \
  > logs/main_gpu_3000_seed2026.log 2>&1 &
echo $! | tee logs/main_gpu_3000_seed2026.pid
```

重新登录后检查进程。nohup 不提供调度器式自动恢复，以上命令也不自动记录最终退出码；以日志、产物和完成性检查联合判断。

### 6.3 Slurm（替代上一节，不要重复启动）

```bash
cat > main_gpu_3000.sbatch <<'SH'
#!/usr/bin/env bash
#SBATCH --job-name=optest-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
test ! -e outputs/main_gpu_3000_seed2026
python -u scripts/train.py --config configs/main_gpu_3000_seed2026.yaml
SH
mkdir -p logs
sbatch main_gpu_3000.sbatch
```

32G 和 24 小时只是作业模板，不是运行资源保证。按管理员要求增加 partition/account/qos；有的集群使用 `--gpus`，按当地规范修改。不要在作业内覆盖调度器分配的 CUDA_VISIBLE_DEVICES。

## 7. 监控、中断与恢复

### 7.1 监控

```bash
tail -n 80 logs/main_gpu_3000_seed2026.log
nvidia-smi
ls -lh outputs/main_gpu_3000_seed2026/
stat outputs/main_gpu_3000_seed2026/latest.pt
# 普通后台作业：
ps -p "$(cat logs/main_gpu_3000_seed2026.pid)" -o pid,etime,%cpu,%mem,cmd
# Slurm 作业：
squeue -u "$USER"
# JOB_ID 替换为实际作业号：
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

当前主入口不逐 episode 打印进度，训练 CSV 主要在循环结束写入。空日志不一定卡死，结合 CPU、GPU、进程和 checkpoint 修改时间判断。初始验证期间可能尚无 checkpoint。

不能并发写同一输出目录。不要反复读取正在写入的 checkpoint 监控进度：主模型保存尚未采用原子替换。

### 7.2 停止与备份

核对进程后，普通作业使用 `kill -TERM PID`，Slurm 使用 `scancel JOB_ID`。当前没有保证退出前自动保存的处理；可能留下不完整的最新 checkpoint。不要首先 `kill -9`，不要批量结束他人的 Python。

进程退出后，备份整个运行目录到新名称，再讨论恢复。建议先人工确认空间和目标名称，例如：

```bash
# 仅在旧进程已停止、备份目标不存在时执行：
test ! -e outputs/main_gpu_3000_seed2026_backup_before_resume && \
  cp -a outputs/main_gpu_3000_seed2026 outputs/main_gpu_3000_seed2026_backup_before_resume
```

### 7.3 恢复不是新预算微调

现有恢复命令：

```bash
python -u scripts/train.py --config configs/main_gpu_3000_seed2026.yaml \
  --resume outputs/main_gpu_3000_seed2026/latest.pt
```

前提：旧进程停止、有备份、checkpoint 可读取、配置／阶段／划分不变。只加载可信的自己生成的文件。

当前恢复包含模型、优化器、学习率调度器、global_step 和历史最佳值，但不完整恢复随机状态、采样器位置、早停计数和训练历史。恢复后可能覆盖训练 CSV，无法保证精确重放，也可能导致统一比较预算检查失败。

**严格复现的长任务应先实施第 14 节的恢复和日志改造。** 如必须用当前功能，保留恢复区间和历史文件，审计时合并且不重复计数。

不要用旧 150-episode best.pt 的 `--resume` 开始新 3000-episode 实验：旧调度器和阶段偏移也会恢复。本文的新预算与扩容均推荐从头训练；仅权重微调需要新增专门入口。

## 8. 完整评估与可视化

### 8.1 完成性检查

针对本手册的全新原规模 3000-episode 训练：

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
root = Path('outputs/main_gpu_3000_seed2026')
for name in ['best.pt', 'latest.pt', 'train_log.csv', 'validation_log.csv', 'summary.json', 'per_scene.csv']:
    assert (root / name).is_file(), name
summary = json.loads((root / 'summary.json').read_text())
assert summary['num_scenarios'] == 108 and summary['num_base_dags'] == 54
assert summary['valid_schedule_rate'] == 1.0
log = pd.read_csv(root / 'train_log.csv')
trained = log[log['phase'].isin(['low_pretrain', 'high_train', 'joint'])]
assert len(trained) == 3000, f'Budget incomplete: {len(trained)}'
assert trained['step'].is_unique
assert set(trained['step']) == set(range(1, 3001))
print('Episodes:', len(trained))
print('Unique sampled scenarios:', trained['scenario_id'].nunique(), '/ 864')
print('Selected model mean ratio:', summary['mean_ratio'])
PY
```

如改了预算或允许早停，应调整验收目标并如实披露实际预算。不能把旧报告存在当成本次完成；检查时间戳。有放回采样意味着 3000 episodes 不保证 864 场景全覆盖，场景覆盖率也不等于基础 DAG 覆盖率。

### 8.2 独立 CPU 评估配置

采用新目录，避免修改训练产物。只改变评估设备，不改 checkpoint：

```bash
python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('outputs/main_gpu_3000_seed2026/config.yaml').read_text(encoding='utf-8'))
config['device'] = 'cpu'
config['output_dir'] = 'outputs/main_gpu_3000_seed2026_eval_cpu'
assert not Path(config['output_dir']).exists()
with Path('configs/main_gpu_3000_seed2026_eval_cpu.yaml').open('x', encoding='utf-8') as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for policy in heft hrl hrl_safe portfolio; do
  python scripts/evaluate.py \
    --config configs/main_gpu_3000_seed2026_eval_cpu.yaml \
    --policy "$policy" --checkpoint outputs/main_gpu_3000_seed2026/best.pt \
    --split validation || break
done
```

确保四次评估全部成功后再继续。重复运行同一 policy 会覆盖独立评估目录的对应结果，先备份需保留的版本。

HEFT 和 portfolio 不使用传入的神经 checkpoint。portfolio 是默认扰动组合，不是 beam + blocks。GPU 推理时间应另建配置／目录测量，不与 CPU 时间作为同硬件成绩混排。

### 8.3 结果一致性检查

```bash
python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
root = Path('outputs/main_gpu_3000_seed2026_eval_cpu')
reference = None
for policy in ['heft', 'hrl', 'hrl_safe', 'portfolio']:
    directory = root / f'eval_{policy}_validation'
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['num_scenarios'] == 108 and summary['num_base_dags'] == 54
    assert summary['valid_schedule_rate'] == 1.0
    frame = pd.read_csv(directory / 'per_scene.csv').set_index('scenario_id').sort_index()
    assert len(frame) == 108 and frame.index.is_unique
    assert np.isfinite(frame['ratio']).all()
    assert (frame['heft_makespan'] > 0).all()
    if reference is None:
        reference = frame
    else:
        assert frame.index.equals(reference.index)
        assert np.allclose(frame['heft_makespan'], reference['heft_makespan'], rtol=1e-10, atol=1e-10)
    if policy in ['hrl_safe', 'portfolio']:
        assert (frame['ratio'] <= 1 + 1e-9).all()
    print(policy, summary['mean_ratio'], summary['max_ratio'])
PY
```

### 8.4 绘图

```bash
for policy in heft hrl hrl_safe portfolio; do
  python scripts/plot_formal_results.py \
    --evaluation-dir "outputs/main_gpu_3000_seed2026_eval_cpu/eval_${policy}_validation" || break
done
python scripts/plot_policy_comparison.py \
  --output-dir outputs/main_gpu_3000_seed2026_eval_cpu --split validation
```

通用比较图会跳过不存在的目录，成功出图不代表所有方法都评估了。检查图例：这里应有四种策略，不会自动加入 Graph PPO、Tier MAPPO、beam + blocks。

训练曲线位于训练输出目录；评估图位于相应评估目录。服务器无需桌面，将图片下载查看。通用报告的 mean_ratio_ci95 是场景级 bootstrap，不能称为基础 DAG 分组置信区间；历史六方法的配对统计才按 DAG 分组。

## 9. 扩大训练数据

### 9.1 实验定义

先原规模充分训练，再加 500，再考虑 1000。不要一开始启用几千任务全部数据。

本节给出当前主模型入口可运行的方案：保留原三份划分，新规模全部加入 train。它只回答“大图训练是否改善原 108 场景”，**不提供大图独立泛化结论**。如要预留新增规模的独立评估数据，必须训练前另行冻结协议，不能事后将训练过的数据划为测试。

### 9.2 创建扩展 manifest

确认 500 的同构、异构数据包均存在。以下命令加载数据但不运行 test 评估：

```bash
python - <<'PY'
from pathlib import Path
from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.split import SplitManifest, SplitManager, scenario_base_key
root = 'data/raw/zenodo-18927122-derived'
source = Path('data/manifests/grapheonrl_mixed_iid_seed7.json')
target = Path('data/manifests/grapheonrl_train500_fixedval108_seed7.json')
assert source.is_file() and not target.exists()
old = SplitManifest.read(source)
scenarios = load_scenarios(default_registry(root, [50, 100, 300, 500]), {'grapheonrl': root})
known = set(old.train_base_dag_ids) | set(old.validation_base_dag_ids) | set(old.test_base_dag_ids)
loaded = {scenario_base_key(item) for item in scenarios}
assert known <= loaded, 'Original DAGs missing'
added = loaded - known
assert added, 'No new DAGs'
assert all(item.num_tasks == 500 for item in scenarios if scenario_base_key(item) in added)
old_view = SplitManager.apply([item for item in scenarios if scenario_base_key(item) in known], old)
expanded = SplitManifest(
    train_base_dag_ids=tuple(sorted(set(old.train_base_dag_ids) | added)),
    validation_base_dag_ids=old.validation_base_dag_ids,
    test_base_dag_ids=old.test_base_dag_ids,
    seed=old.seed,
    protocol='expanded_train_fixed_validation',
    resource_seeds=old.resource_seeds,
    metadata={**old.metadata, 'num_base_dags': len(loaded),
              'parent_manifest': str(source), 'added_train_base_dags': len(added),
              'validation_unchanged': True, 'added_scales_train_only': [500]},
)
splits = SplitManager.apply(scenarios, expanded)
identity = lambda values: sorted((item.scenario_id, scenario_base_key(item)) for item in values)
assert identity(splits['validation']) == identity(old_view['validation'])
assert identity(splits['test']) == identity(old_view['test'])
assert len(splits['validation']) == 108
assert len({scenario_base_key(item) for item in splits['validation']}) == 54
assert len(splits['test']) == 108
expanded.write(target)
for name, values in splits.items():
    print(name, len(values), 'scenarios', len({scenario_base_key(item) for item in values}), 'DAGs')
print('Wrote', target)
PY
```

该检查按来源＋基础 DAG 身份防泄漏，不是通用图同构去重。引入其他来源或重命名数据时需补充内容／拓扑重复审计。

不要使用当前 create_protocol.py 重新随机划分替代此流程：它默认加载原三个规模，重新切分也破坏历史验证可比性。

### 9.3 创建扩容训练配置

第一轮保持 3000 episodes，便于作 episode 预算对照；大图每回合决策更多，仍不代表完全等计算量。

```bash
python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('configs/main_gpu_3000_seed2026.yaml').read_text(encoding='utf-8'))
name = 'main_gpu_train500_3000_seed2026'
config['experiment']['name'] = name
config['dataset']['archive_task_counts'] = [50, 100, 300, 500]
config['dataset']['split_manifest'] = 'data/manifests/grapheonrl_train500_fixedval108_seed7.json'
config['curriculum']['stages'] = [
    {'min_tasks': 50, 'max_tasks': 100, 'episodes': 600},
    {'min_tasks': 50, 'max_tasks': 300, 'episodes': 600},
    {'min_tasks': 50, 'max_tasks': 500, 'episodes': 1800},
]
config['output_dir'] = f'outputs/{name}'
assert Path(config['dataset']['split_manifest']).is_file()
assert not Path(config['output_dir']).exists()
with Path(f'configs/{name}.yaml').open('x', encoding='utf-8') as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
```

首次先做真实 500 图短跑，不能随机抽到小图后就宣布大图可运行：

```bash
python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('configs/main_gpu_train500_3000_seed2026.yaml').read_text(encoding='utf-8'))
name = 'main_gpu_train500_smoke_seed2026'
config['experiment']['name'] = name
config['training'].update(high_train_episodes=1, joint_train_episodes=1, evaluation_interval=1)
config['curriculum']['stages'] = [{'min_tasks': 500, 'max_tasks': 500, 'episodes': 2}]
config['output_dir'] = f'outputs/{name}'
assert not Path(config['output_dir']).exists()
with Path(f'configs/{name}.yaml').open('x', encoding='utf-8') as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --config configs/main_gpu_train500_smoke_seed2026.yaml
# 短跑成功并检查完整 validation 后，另行启动正式训练：
# CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --config configs/main_gpu_train500_3000_seed2026.yaml
```

长任务用第 6 节 nohup 或 Slurm 模板替换配置和日志名。当前采样器不支持配置“大图固定占 20%”；这里仅扩大可选范围，不保证各规模比例。

### 9.4 扩容后评估

重复第 8 节，把路径中的 main_gpu_3000_seed2026 替换为 main_gpu_train500_3000_seed2026。评估配置保留扩容 manifest 和 archive_task_counts，validation 仍是原 108。

覆盖率分母改成第 9.2 节实际训练场景数，不再写死 864。比较扩容前后 scenario ID、HEFT 分母和数据 hash。

如扩至 1000，要更换唯一文件名／输出名，显式修改规模列表、新增规模断言、manifest metadata、课程范围，再重复短跑。不要将 archive_task_counts 设为 null 来“全用上”：解除筛选可能纳入 FPPPP 等不同数据，原 manifest 不适用。

## 10. 多种子与图学习基线

### 10.1 主模型多种子

单种子稳定后，按如下方式生成 2027、2028，固定 manifest 不变：

```bash
python - <<'PY'
from copy import deepcopy
from pathlib import Path
import yaml
base = yaml.safe_load(Path('configs/main_gpu_3000_seed2026.yaml').read_text(encoding='utf-8'))
for seed in [2027, 2028]:
    config = deepcopy(base)
    name = f'main_gpu_3000_seed{seed}'
    config['experiment'] = {'name': name, 'seed': seed}
    config['output_dir'] = f'outputs/{name}'
    assert not Path(config['output_dir']).exists()
    with Path(f'configs/{name}.yaml').open('x', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
PY
```

普通两卡服务器可在两个终端分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --config configs/main_gpu_3000_seed2027.yaml
# 另一终端，先确认 GPU 1 可使用：
CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --config configs/main_gpu_3000_seed2028.yaml
```

这是一卡一独立模型，不是 DDP；Slurm 分别申请资源。各 seed 按第 8 节独立评估，报告每个 seed 均值和跨 seed 标准差，不把重复验证场景都当作独立 DAG。

### 10.2 图学习基线

当前固定协议可以直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_graph_baselines.py \
  --config configs/zenodo_graph_ppo_comparison.yaml \
  --device cuda --seed 2026 --epochs 3 --output outputs/graph_ppo_gpu_seed2026

CUDA_VISIBLE_DEVICES=0 python -u scripts/train_graph_baselines.py \
  --config configs/zenodo_tier_mappo_comparison.yaml \
  --device cuda --seed 2026 --epochs 3 --output outputs/tier_mappo_gpu_seed2026
```

输出必须为新目录。3 epochs 对原 864 场景是 2592 episodes，不等于主模型 3000。先定义是按遍历、决策数还是计算时间匹配预算，再作公平对比。

**当前基线 load_fixed_splits 硬编码 [50,100,300]、原 manifest 文件名、864/108/108 场景数。它不能直接接受扩容配置。** 需要先改造和测试入口；扩容主模型与旧基线仅可作为不同训练条件的描述性比较。

## 11. 六方法比较的边界

### 11.1 历史六方法复核

此命令复核旧结果，不是新扩容模型比较。需要旧模型旁的 train_log.csv、原 manifest 和数据：

```bash
python scripts/compare_main_models.py \
  --main-checkpoint outputs/zenodo_heft_safe_fast_2026/best.pt \
  --graph-checkpoints \
    outputs/graph_baselines_cpu_20260917/graph_ppo/seed_2026/best.pt \
    outputs/graph_baselines_cpu_20260917/tier_mappo/seed_2026/best.pt \
  --search-config configs/zenodo_heft_safe_search_beam3_r3_blocks_2026.yaml \
  --output outputs/main_model_comparison_linux_recheck
python scripts/plot_main_model_comparison.py \
  --comparison outputs/main_model_comparison_linux_recheck
```

入口固定 CPU、Torch 单线程、seed 2026。输出 comparison.csv、per_scene_all.csv、paired_statistics.json、status.json 和图表。总记录为 648，每方法 108 场景／54 DAG。输出目录不能重用。

### 11.2 新实验使用限制

当前比较器要求两图基线完整训练、seed=2026、预算与 transitions 一致；主模型日志显示完成全部配置预算；数据／资源／归一化协议兼容。早停、恢复后日志缺失、扩容和其他 seed 均可能被拒绝。

原规模 3000-episode 主模型若符合检查，可以用于数值比较，但当前绘图脚本生成的 figures/RESULTS.md 仍写死“历史主模型 150 episodes、最佳 step=100”等旧叙述。新实验必须先将报告改成动态读取真实元数据，或人工修订核验，不能直接引用旧文字。

扩容与多种子需要泛化入口：分别记录训练协议，仍严格校验固定验证身份与资源。不要通过删除校验、伪造配置 hash、把新 manifest 改成旧名字来绕过限制。

## 12. 常见问题及解决办法

| 现象 | 原因检查 | 解决办法 |
| --- | --- | --- |
| nvidia-smi 无设备 | 是否计算节点／分配到 GPU | 申请 GPU 节点；请管理员检查驱动 |
| CUDA available=False | Python 环境、Torch 构建、设备可见性 | 激活正确环境；按官方选择器安装兼容 wheel；检查作业资源 |
| no kernel image | GPU 架构不被当前构建支持 | 选择支持该 GPU 的 PyTorch 构建，不随机轮换 CUDA 包 |
| driver version insufficient | runtime 与驱动不兼容 | 管理员升级驱动或改用受支持构建，不擅自改共享驱动 |
| CUDA out of memory | 其他进程、大图、batch、轨迹 | 降 batch，先小图，检查资源；轨迹内存过大需要实现改造，减 batch 未必解决 |
| 只有 Killed，无 Python 报错 | 主机内存或作业内存限额 | 看 free、MaxRSS；减少同时加载规模或申请更多内存，后续考虑懒加载 |
| illegal memory access | 首个 CUDA 错误、cuDNN 配置 | 退出故障进程，新目录重试短跑；保留禁用 cuDNN；同步调试定位，不能继续复用错误上下文 |
| GPU 利用率低 | CPU 模拟、逐步小模型推理 | 先剖析，不能断言 GPU 无效；批量推理／并行环境需改代码 |
| 日志不刷新／无 train_log.csv | 当前日志多在结束写入 | 结合进程、资源、latest.pt 时间判断；初始验证可能尚无 checkpoint |
| ModuleNotFoundError | 未激活环境／未安装项目 | 在根目录 editable install，设置 PYTHONPATH |
| base DAG absent from manifest | 数据规模新增而 manifest 未变 | 按第 9 节扩展训练划分，不能重划原验证 |
| validation 不是 108 | 数据缺失、limit、manifest、资源改变 | 停止并恢复固定协议，不通过截取前 108 个补救 |
| fixed split mismatch | 用固定基线入口运行扩容协议 | 先改造入口，不伪装原 manifest |
| 输出已存在 | 重用了历史目录 | 新建独立配置和输出；主训练未必主动阻止覆盖 |
| checkpoint 尺寸不匹配 | 模型配置／观察维度／类别不一致 | 使用训练对应模型配置，不能互换 HRL 与图基线权重 |
| checkpoint ZIP/read 错误 | 上传或保存中断 | 对比 hash，使用完整备份；不要反复加载损坏文件 |
| resume 后立即结束 | global_step 已达到配置预算 | 检查恢复含义，变预算微调不等于 resume |
| 恢复后训练日志变短 | 历史未恢复且被覆写 | 使用备份审计区间，改造恢复；不虚填预算 |
| protocol/hash mismatch | manifest 字节、嵌入路径或配置变化 | 保留原文件和相对布局；原绝对路径需专门迁移适配，不伪造 hash |
| Matplotlib 显示错误 | 无桌面后端 | 设置 MPLBACKEND=Agg；缺字体另行安装，不把警告当作训练失败 |
| bash\r、^M | Windows CRLF | 只对相关 shell 文件转换 LF，不改数据和 checkpoint |
| 加大图后变差 | 采样比例、课程、决策预算不同 | 对照原规模实验，统计覆盖；不更改验证集制造提升 |
| ratio 一直约 1 | 初始 HEFT 等价模型仍最佳 | 查 checkpoint step、KL、loss 和残差约束，不等于代码没运行 |
| 期待必达 0.8 | 混淆预算增大与算法收益 | 无保证；先验证覆盖、多种子稳定性，再设计算法改进 |

CUDA 同步定位命令示意：

```bash
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/train.py --config configs/YOUR_NEW_DEBUG_CONFIG.yaml
```

YOUR_NEW_DEBUG_CONFIG.yaml 是占位符，先复制创建唯一输出目录的调试配置。同步调试不是默认性能配置。非法访问后终止并重启 Python。

网络受限：在兼容 Linux/Python 平台准备 wheelhouse 和依赖，再传入服务器安装，不使用 Windows wheel。数据和依赖齐备后训练可离线运行。

## 13. 归档与最终验收

每次实验保存：源码版本／快照、执行命令、起止时间、实际配置、manifest、环境信息、checkpoint 与数据 hash、完整日志、逐场景指标、图表。记录训练 episodes、决策数、访问覆盖、seed、设备、线程、是否恢复和实际退出状态。

建议目录：

```text
outputs/main_gpu_3000_seed2026/           原规模训练
outputs/main_gpu_3000_seed2026_eval_cpu/  独立评估
outputs/main_gpu_train500_3000_seed2026/  扩容训练
logs/                                   作业日志
environment/                            软件与硬件信息
configs/main_gpu_*.yaml                  实际配置
data/manifests/                         原始与扩展划分分开保存
```

只归档已停止写入的 checkpoint。没有 Git 时保存源码快照和 hash，不声称有不存在的提交。只加载可信 checkpoint；部分入口 torch.load 使用 weights_only=False，任意下载的 .pt 不是安全数据文件。

验收清单：

- [ ] GPU 小测试、项目测试、真实 PPO smoke 通过。
- [ ] 原始文件完整，固定 manifest 存在且可核验。
- [ ] 所有验证为原 108 场景／54 DAG，无 proxy、无 test 评估。
- [ ] 训练正常结束，实际预算与日志一致。
- [ ] 历史产物／失败日志未被覆盖。
- [ ] 有效调度率 1.0，ratio 和 HEFT 分母可复算。
- [ ] 图表方法数、预算、设备、样本量与报告一致。
- [ ] 神经主模型、安全组合、独立搜索分开解释。
- [ ] 扩容不冒充大图独立泛化；单种子不冒充稳健结论。
- [ ] 恢复、早停、不同训练预算等限制如实披露。

## 14. 建议优先实施的工程改造

以下功能尚未因本文而实现，建议按风险顺序推进：

1. **可靠保存和恢复**：原子 checkpoint，完整随机／采样器／早停状态，历史日志恢复，配置和 manifest 校验。
2. **增量日志**：定期落盘步数、阶段、覆盖、耗时和显存，增加状态文件及退出检查。
3. **仅权重初始化**：独立 init-from，重置优化器、调度器与计数，完整初始验证。
4. **可控覆盖采样**：规模和资源类型分层、组内无放回，记录基础 DAG 与场景访问次数。
5. **扩容比较支持**：区分训练协议与固定验证协议，保留防泄漏和身份校验。
6. **动态报告**：从产物读取预算、最佳步数、种子、是否恢复，移除写死历史文字。
7. **剖析后加速**：先测模拟／推理／更新耗时，再做缓存、批量推理、并行采样或懒加载；最后考虑 DDP。

建议先完成 GPU smoke，再决定是否先实施 1–2 后启动长任务；原规模稳定后再扩容。扩大训练量不保证达到 ratio=0.8。

## 附录：文档验证范围

编写时已检查全部 11 段内嵌 Python 示例的语法，并在隔离临时目录实际执行 6 段配置生成示例，确认产生的 8 个新配置关闭 proxy、没有数据子集限制、课程总步数与训练预算一致；三个相对文档链接均有效。扩展 manifest 的脚本只做了语法与 API 核对，未在本文编写过程中加载新增规模并执行划分生成。没有启动训练，也没有在实际 Linux GPU／Slurm 服务器执行命令，服务器端验收仍须按本文逐项完成。

## 附录：代码与资料索引

| 功能 | 文件 |
| --- | --- |
| 主训练入口 | `scripts/train.py` |
| checkpoint 保存与加载 | `src/cpn_hrl_dag/algorithms/hierarchical_trainer.py` |
| 采样器 | `src/cpn_hrl_dag/datasets/sampling.py` |
| 划分与防泄漏 | `src/cpn_hrl_dag/datasets/split.py` |
| 原生数据与规模筛选 | `src/cpn_hrl_dag/datasets/grapheonrl_adapter.py` |
| 固定协议基线 | `src/cpn_hrl_dag/experiments/graph_baselines.py` |
| 单策略评估 | `scripts/evaluate.py` |
| 通用绘图 | `scripts/plot_formal_results.py`、`scripts/plot_policy_comparison.py` |
| 六方法比较 | `scripts/compare_main_models.py` |
| 六方法图表与报告 | `scripts/plot_main_model_comparison.py` |

补充：[主模型架构与流程](主模型架构与流程.md)、[原策略对比说明](云边端多种策略对比.md)。
