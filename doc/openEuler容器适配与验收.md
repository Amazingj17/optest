# openEuler 容器适配与验收

本项目提供 openEuler CPU 验收入口，复用现有模型、训练器、环境、动作 mask 和评估器。
容器实测与未来虚拟机实测分别记录；Docker Desktop / WSL2 的宿主内核不是 openEuler 内核。
本轮不涉及 GPU、ARM/鲲鹏、昇腾，也不把容器结果当作这些平台的适配证据。

## 1. 适配范围

| 项目 | 配置 |
| --- | --- |
| 基础镜像 | `openeuler/openeuler:24.03-lts-sp4`，Dockerfile 固定镜像 digest |
| 架构 | x86_64 |
| Python | 镜像内 Python 3.11；项目最低要求 Python 3.10 |
| PyTorch | `2.9.1+cpu`，官方 CPU wheel |
| 其他依赖 | `deploy/openeuler/requirements.txt`；运行时另存完整解析版本 |
| 绘图 | `MPLBACKEND=Agg`，无桌面环境 |
| CPU 线程 | OMP/MKL/OpenBLAS 各 1，短训练 Torch 线程为 1 |
| 数据 | 原固定划分，训练 864 场景，validation 108 场景 / 54 基础 DAG |
| 权限与网络 | 数据和既有模型只读挂载；完成镜像构建后可断网验收 |

适配文件：

- `deploy/openeuler/Dockerfile`：安装系统依赖、CPU PyTorch 和项目。
- `.dockerignore`：阻止原始数据、模型、历史输出、缓存和本地配置进入构建上下文。
- `configs/openeuler_cpu_smoke.yaml`：1 次高层更新阶段、1 次联合更新阶段的小模型短训练。
- `scripts/verify_platform.py`：识别实际系统、运行测试/训练/验证、保存证据。
- `tests/conftest.py`：只在 Windows 上启用原有权限兼容逻辑；Linux 保留 pytest 默认临时目录权限。
- `tests/test_platform_acceptance.py`：检查非目标系统拒绝、容器身份、指标核验、失败与防覆盖行为。

## 2. 验收逻辑

一条命令依次完成：

1. 保存真实 `/etc/os-release`、内核、架构、Python、依赖版本和源码 SHA256。
2. 拒绝非 openEuler 系统；检测到容器时禁止标记为 VM/native。
3. `pip check`、本地数据检查、7 个数据归档的 SHA256。
4. 完整 `pytest`，输出控制台日志及 JUnit XML；任何失败中止后续步骤。
5. 运行现有 `train_main_comparison.py`，实际执行两个 episode，验证高低层均产生优化器更新。
6. 保存并在新进程加载 `best.pt`，核对 108 场景逐项 makespan / HEFT 分母与训练结束评估一致。
7. HEFT 同场景评估，检查合法率 100% 和自身比值 1。
8. 可选：只读加载已训练的正式 HRL checkpoint，执行固定 validation 评估，核对权重未被修改。

每份结果检查 108 个唯一场景、54 个 DAG、100% 合法率、有限且为正的 makespan/ratio，
并从逐场景 CSV 重算均值。所有阶段完成后才记录 `status: passed`；失败记录错误和 traceback。
输出目录必须不存在，避免覆盖先前证据。脚本只评估 validation，不评估 test；
加载固定划分时会检查 test 分组身份和数量，这不是 test 性能评测。

**短训练仅证明可训练、可保存与可加载，不说明模型收敛，也不替代三种子正式训练。**
两次训练后选中的 best 可以是 step 0；实际发生更新由 `budget.json` 和优化器计数证明。

## 3. Windows / Docker Desktop 运行

在项目根目录执行 PowerShell。先按 [数据准备说明](../data/README.md)准备完整的 50/100/300
同构及异构归档。Dockerfile 不会打包这些数据或 `outputs/` 中的权重。

```powershell
docker build -f deploy/openeuler/Dockerfile -t optest:openeuler-24.03-cpu .

$repoRoot = (Get-Location).Path
$evidenceRoot = Join-Path $repoRoot 'outputs/platform'
New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null

docker run --rm --network none --cpus 2 --memory 4g --pids-limit 256 `
  --mount "type=bind,source=$repoRoot/data/raw,target=/opt/optest/data/raw,readonly" `
  --mount "type=bind,source=$evidenceRoot,target=/evidence" `
  optest:openeuler-24.03-cpu `
  python scripts/verify_platform.py `
  --output /evidence/openeuler_container_run01 --execution-kind container
```

再次验收时换一个新的输出目录名，不删除旧证据。构建阶段需要访问 openEuler 软件源、PyPI 和
PyTorch 官方 CPU wheel 源；运行阶段 `--network none` 禁用网络。

要同时检查最新 seed2026 正式权重，在 `docker run` 的镜像名前增加：

```powershell
--mount "type=bind,source=$repoRoot/outputs/gpu/outputs/multiseed_20260925_122243/all_models_seed2026,target=/frozen,readonly"
```

在 `verify_platform.py` 参数末尾增加：

```text
--frozen-config /frozen/configs/residual_hrl.yaml
--frozen-checkpoint /frozen/training/residual_hrl/best.pt
```

两项必须同时提供。它们是原训练配置与对应权重；无需修改原配置中的 GPU 设备或服务器输出路径，
验收入口会在新的输出目录生成 CPU 评估配置，不改原文件。仅加载自己训练或可信来源的 checkpoint。

## 4. Linux Docker 运行

```bash
docker build -f deploy/openeuler/Dockerfile -t optest:openeuler-24.03-cpu .
mkdir -p outputs/platform
docker run --rm --network none --cpus 2 --memory 4g --pids-limit 256 \
  --mount "type=bind,source=$PWD/data/raw,target=/opt/optest/data/raw,readonly" \
  --mount "type=bind,source=$PWD/outputs/platform,target=/evidence" \
  optest:openeuler-24.03-cpu \
  python scripts/verify_platform.py \
  --output /evidence/openeuler_container_run01 --execution-kind container
```

## 5. 后续迁移到 openEuler 虚拟机

以下是待执行步骤，不是已经完成的 VM 实测。建议先安装相同的 openEuler 24.03 LTS-SP4 x86_64。
在 VM 内操作，而不是在 Windows 宿主机或另一个容器里操作：

```bash
sudo dnf install -y python3-pip git libgomp
python3 --version
git clone https://github.com/Amazingj17/optest.git
cd optest
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.3
python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r deploy/openeuler/requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

放置原始数据；需要复现正式模型时同时拷贝对应配置和 checkpoint。然后执行：

```bash
python scripts/verify_platform.py \
  --output outputs/platform/openeuler_vm_run01 --execution-kind vm
```

脚本记录的 `execution_kind=vm` 是操作者声明；它会排除明确的容器标记，但不会自动证明虚拟化产品。
VM 交付材料还应附安装过程、发行版/内核信息、虚拟机配置、实际开发调试记录和运行录屏。
不要求在 VM 上重做全部三种子长训练，也不能将 Windows/Ubuntu 历史训练改称 openEuler 训练。

## 6. 产物与提交边界

```text
outputs/platform/<run>/
  acceptance.json              # 总状态、阶段命令/退出码/时间、验收指标
  environment.json             # 发行版、内核、架构、Python、完整依赖版本
  os-release.txt
  requirements-resolved.txt
  source_sha256.json
  data_sha256.json
  pytest.xml
  logs/
  smoke_config.yaml
  smoke_training/              # checkpoint、实际训练预算、日志和 108 场景结果
  smoke_replay/                # 新进程重载后的评估
  heft/
  frozen/                      # 提供正式 checkpoint 时生成
  failure.txt                  # 仅失败时生成
```

`outputs/` 保持 Git 忽略；轻量实测报告与必要证据放到 `docs/platform/`，不要上传权重和数据。
完整解析依赖文件用于追溯版本；重建 CPU 环境时仍需指定 PyTorch CPU 索引，
系统 RPM 软件源也可能更新，因此不宣称 Docker 构建可达到逐字节重现。

容器实测能够证明 openEuler 用户态下的安装、训练和评估可运行。
是否满足赛方“基于国产操作系统开发”的全部认定要求，应结合赛方口径与后续 VM 开发证据说明。
