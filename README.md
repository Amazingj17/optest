# CPN-HRL-DAG：云—边—端 DAG 任务调度

## 项目简介

面向云、边缘和终端异构计算资源的 DAG（有向无环图）任务调度实验框架。项目在统一仿真环境中实现分层强化学习、启发式搜索与基线算法，目标是在满足任务依赖和资源约束的前提下缩短工作流总完成时间（makespan），并提供可复现的训练、评测和离线可视化演示。

项目采用 Python 实现，可安装的包名为 `cpn-hrl-dag`，当前版本为 `0.1.0`。本 README 位于项目根目录；根目录的 `src/` 是 Python 代码项目目录，其中的 `src/cpn_hrl_dag/` 是包源码。

## 应用场景

| 场景 | 使用方式 |
| --- | --- |
| 云—边—端协同调度研究 | 将具有依赖关系的任务映射到异构资源，比较计算能力与通信条件对完成时间的影响 |
| 工作流调度算法对比 | 在相同场景、资源与数据划分下评测 HEFT、强化学习和搜索策略 |
| 强化学习实验与教学 | 观察任务选择、节点分配、动作掩码与分阶段训练过程 |
| 项目展示与结果复核 | 使用内置场景和权重生成离线网页，查看调度方案及实验指标 |
| 操作系统兼容性验证 | 在 openEuler、openKylin 等环境中验证安装、测试、训练与模型重载流程 |

## 系统环境

| 项目 | 要求或已验证环境 |
| --- | --- |
| Python | 包声明要求 Python ≥ 3.10，建议使用 Python 3.11 或 3.12 |
| 开发运行环境 | 提供 Windows 启动脚本；Linux 平台验收记录覆盖 openEuler 24.03 LTS-SP4、openKylin 2.0 SP2（x86_64） |
| CPU / GPU | 离线演示和冒烟训练可使用 CPU；`device: cuda` 的配置需要支持 CUDA 的 GPU、匹配的驱动与 PyTorch |
| 浏览器 | 查看本地 HTML 结果需要支持 JavaScript 和 SVG 的现代浏览器 |
| 文件系统 | 数据目录可读，实验输出目录可写；完整数据与多次实验需预留相应存储空间 |
| 网络 | 首次安装依赖和下载完整数据需要网络；资源准备完成后可离线运行演示 |
| 可选构建工具 | Python 分发包构建使用 `build` 与 setuptools；容器构建需要 Docker |

CPU 冒烟训练仅用于检查流程。完整训练和搜索的时间、内存需求随 DAG 规模、模型和候选数量变化，项目未给出统一的最低硬件配置。macOS 可参考安装命令，但仓库未提供该平台的验收记录。

## 主要功能

- **统一调度仿真**：建模任务执行时间、跨节点通信带宽与延迟，支持插入式调度、可行动作掩码和调度合法性检查。
- **分层强化学习**：高层 LSTM 策略选择就绪任务，低层 GAT 策略选择计算节点，使用 PPO 训练；支持启发式残差策略和分阶段训练。
- **基线与搜索**：提供 HEFT、Random、Greedy EFT、Graph PPO、Tier MAPPO，以及结合候选策略、束搜索和局部搜索的 HEFT-safe 策略。
- **独立混合策略**：HEFT、搜索和已训练 HRL 分别生成完整调度，校验后选择 makespan 最小的方案。HRL 候选独立于搜索分支。
- **复现实验**：固定数据划分、随机种子、配置哈希与运行环境记录，输出逐场景指标、汇总报告和训练曲线。
- **离线展示**：内置 6 个场景和模型权重，可在 CPU 上运行并生成本地 HTML 结果页面。

## 依赖要求

依赖声明以 [src/pyproject.toml](src/pyproject.toml) 为准：

| 类别 | 依赖 | 用途 |
| --- | --- | --- |
| 核心依赖 | NumPy ≥ 1.24、NetworkX ≥ 3.0、PyYAML ≥ 6.0 | 数值计算、DAG 处理和配置读取 |
| `train` | PyTorch ≥ 2.0 | 模型推理、PPO 与训练 |
| `analysis` | pandas ≥ 2.0、Matplotlib ≥ 3.7、SciPy ≥ 1.9 | 结果分析、绘图和统计 |
| `dev` | pytest ≥ 7.0 | 自动测试 |
| 包构建 | setuptools ≥ 68；使用构建命令时另安装 `build` | 生成 wheel 与源码分发包 |

离线演示、训练和完整测试建议安装 `train,analysis,dev` 三组依赖。[demo/requirements.txt](demo/requirements.txt) 记录演示的固定版本，平台验收目录保留各次运行的依赖版本。

如需复现演示依赖，可在兼容这些版本的 Python 环境中执行 `python -m pip install -r demo/requirements.txt`。固定版本的 Python 要求可能高于项目包声明的最低版本；CUDA 训练应按实际驱动环境选择 PyTorch 分发版本。

## 构建方法

本项目为 Python 源码项目，开发运行可直接采用下一节的可编辑安装。需要生成可分发安装包时，在已激活的虚拟环境中，从**项目根目录**执行：

```bash
python -m pip install build
python -m build ./src --outdir ./dist
```

构建完成后，根目录 `dist/` 中应生成 wheel（`.whl`）和源码分发包（`.tar.gz`）。当前版本的 wheel 文件名为 `cpn_hrl_dag-0.1.0-py3-none-any.whl`。

也可从项目根目录构建 openEuler CPU 容器镜像：

```bash
docker build -f src/deploy/openeuler/Dockerfile -t cpn-hrl-dag:openeuler ./src
```

Docker 构建需要获取基础镜像和依赖。原始数据被构建上下文的忽略规则排除，运行容器验收时应另行挂载到 `/opt/optest/data/raw/`。Dockerfile 默认启动平台验收脚本；具体设置见 [openEuler Dockerfile](src/deploy/openeuler/Dockerfile)。

## 安装方法

### 创建并激活虚拟环境

在项目根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

### 从源码安装

在项目根目录安装项目及训练、分析、测试依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e "./src[train,analysis,dev]"
```

安装后可检查包是否可导入：

```bash
python -c "import cpn_hrl_dag; print('cpn_hrl_dag import OK')"
```

### 安装构建产物

完成上一节的包构建后，也可从项目根目录执行：

```bash
python -m pip install "./dist/cpn_hrl_dag-0.1.0-py3-none-any.whl[train,analysis,dev]"
```

wheel 安装提供 Python 包。本文使用的 `run_demo.py`、实验脚本、配置、演示资产与原始数据仍需从项目目录访问；运行这些入口时应保留源码目录结构。

## 运行方法

### 离线演示

在项目根目录执行：

```bash
python run_demo.py
```

Windows 也可以双击 `run_demo.cmd`。程序校验演示资源的 SHA-256，使用内置模型在 CPU 上运行 HEFT、搜索和 HRL 候选，并在完成后打开结果页面。安装依赖后，演示不需要下载完整数据集或重新训练。

演示覆盖 50、100、300 个任务规模，每个规模包含同构与异构资源场景，共 6 个场景。默认输出到 `demo/runs/<时间戳>/`：

| 文件 | 内容 |
| --- | --- |
| `index.html` | 本地可视化结果页面 |
| `results.json` | 场景、候选方案与调度结果 |
| `summary.json` / `metrics.csv` | 汇总与候选方案指标 |
| `provenance.json` | 权重来源、随机种子、环境和计时口径 |
| `progress.jsonl` / `status.json` | 逐场景进度与运行状态 |

不自动打开浏览器，或指定输出目录：

```bash
python run_demo.py --no-open
python run_demo.py --output demo/runs/my_demo
```

指定目录应为空或尚不存在。只运行一个场景：

```bash
python run_demo.py --scene grapheonrl:rnc50:rand0000:heterogeneous
```

`--scene` 可以重复传入，完整场景 ID 见 [demo/manifest.json](demo/manifest.json)。也可以直接打开已有的 [演示页面](demo/index.html) 或 [四种方法对比页面](demo/verified_comparison/index.html) 查看保存结果；查看页面不会启动新的计算。

### 完整实验的数据准备

完整训练与评测使用 [Zenodo 18927122](https://zenodo.org/records/18927122) 中的 GrapheonRL 数据。原始数据归属 Aasish Kumar Sharma，演示清单记录的数据许可为 **CC-BY-4.0**；使用时应保留原始署名及许可信息。

原始压缩包不随 Git 仓库分发。下载后，按以下结构放置，适配器可直接读取压缩包成员，无需先全部解压：

```text
src/data/raw/zenodo-18927122-derived/
├── system_configs.tar.xz
├── rnc50_homo_json.tar.xz
├── rnc50_hetero_json.tar.xz
├── rnc100_homo_json.tar.xz
├── rnc100_hetero_json.tar.xz
├── rnc300_homo_json.tar.xz
└── rnc300_hetero_json.tar.xz
```

进入代码项目目录后检查本地数据位置：

```bash
cd src
python scripts/prepare_data.py
```

该脚本仅检查目录、系统配置包和工作流压缩包是否存在，不下载数据，也不完整校验全部压缩包内容。默认 50/100/300 任务实验需要上面列出的全部文件。

数据说明见 [src/data/README.md](src/data/README.md)。复现归档结果时保留固定划分文件 [grapheonrl_mixed_iid_seed7.json](src/data/manifests/grapheonrl_mixed_iid_seed7.json)，同一基础 DAG 的资源变体按基础 DAG 分组划分。

### 训练与评测

**以下命令均在根目录下的 `src/` 中运行**，并以完成依赖安装和完整数据准备为前提。配置中的 `data/`、`outputs/` 等相对路径也以该目录为基准。

#### 小规模训练与基线评测

运行默认 CPU 冒烟训练：

```bash
python scripts/train.py --config configs/cpn_hrl_dag.yaml
```

默认配置只运行少量场景和训练 episode，用于验证训练链路。权重与报告写入 `src/outputs/cpn_hrl_dag_smoke/`，包括 `best.pt`、`latest.pt`、训练日志、数据划分和评测报告。

在验证集上评测 HEFT 和训练得到的 HRL：

```bash
python scripts/evaluate.py --config configs/cpn_hrl_dag.yaml --policy heft --split validation
python scripts/evaluate.py --config configs/cpn_hrl_dag.yaml --policy hrl --checkpoint outputs/cpn_hrl_dag_smoke/best.pt --split validation
```

评测结果写入配置输出目录下的 `eval_<策略>_<划分>/`。其他策略包括 `random`、`greedy`、`portfolio`、`search`、`hrl_safe` 和 `hrl_search_safe`；HRL 相关策略需要 `--checkpoint`。模型结构、观测选项与权重应使用匹配的配置。

#### 正式训练与扩展实验

| 入口 / 配置 | 用途 |
| --- | --- |
| `scripts/train_main_comparison.py` | 主模型完整覆盖训练、显式阶段计划与训练诊断 |
| `configs/main_gpu_3000_seed2026.yaml` | CUDA 主模型训练，600 个高层训练 episode + 2400 个联合训练 episode |
| `scripts/train_graph_baselines.py` | Graph PPO / Tier MAPPO 基线训练 |
| `scripts/compare_main_models.py` | 主模型、图基线与搜索在固定完整验证集上的对比 |
| `scripts/evaluate_independent_hybrid.py` | HEFT、搜索、HRL 独立完整调度的混合评测与贡献分析 |
| `scripts/train_dag_pair.py` | DAG-pair 模型及相关蒸馏实验 |
| `configs/generated_smoke.yaml` | 生成资源环境的冒烟实验，DAG 仍依赖原始数据 |

例如，使用可用的 CUDA 环境运行主模型训练：

```bash
python scripts/train_main_comparison.py --config configs/main_gpu_3000_seed2026.yaml
```

图基线示例：

```bash
python scripts/train_graph_baselines.py --config configs/zenodo_graph_ppo_comparison.yaml
python scripts/train_graph_baselines.py --config configs/zenodo_tier_mappo_comparison.yaml
```

运行独立混合评测前，需要将 `configs/hrl_independent_hybrid_seed2026.yaml` 中的 `checkpoint` 和 `search_config` 改为已有实验文件的实际路径；默认值指向本地历史实验，完整实验产物未随仓库分发。

```bash
python scripts/evaluate_independent_hybrid.py --config configs/hrl_independent_hybrid_seed2026.yaml
```

每次新实验建议使用新的 `output_dir`。部分脚本会拒绝覆盖已有非空目录。更多参数可通过相应脚本的 `--help` 查看。

## 指标与已有结果

主要指标为同一仿真环境下相对于 HEFT 的完成时间比值：

```text
ratio = 当前策略 makespan / HEFT makespan
```

`ratio < 1` 表示比 HEFT 更快，`ratio = 1` 表示相当；`mean_ratio` 是逐场景比值的算术平均。评测同时记录合法调度率、优于 HEFT 的比例和推理耗时。搜索策略的推理计时包含候选方案生成与选择，应结合时间开销评估收益。

仓库保留的三种子结果使用相同的 **108 个验证场景 / 54 个基础 DAG**。下表摘自 [mean_std.csv](tests/results/multiseed_20260925_122243/mean_std.csv)，数值保留六位小数：

| 方法 | 三种子平均 `mean_ratio` | 跨种子样本标准差 |
| --- | ---: | ---: |
| HEFT | 1.000000 | 0.000000 |
| Graph PPO | 1.029373 | 0.011873 |
| Tier MAPPO | 1.097279 | 0.017729 |
| HRL搜索 | 0.918223 | 0.000498 |

验证集用于模型选择，因此这些数值属于验证结果。混合策略的大部分收益来自搜索分支，不能全部归因于 HRL。6 场景演示是验证集的固定子集，不能替代完整验证成绩或独立测试结果。完整说明见 [三种子训练验收与结果报告](tests/results/multiseed_20260925_122243/三种子训练验收与结果报告.md)。

归档目录仅保留轻量报告和表格，完整实验权重、训练日志和逐场景调度未随该目录分发；离线演示所需的精选权重单独保存在 `demo/assets/`。

## 测试方法

### 自动化测试

完成依赖安装后，从项目根目录进入 `src/` 执行自动测试。部分数据适配器与集成测试依赖原始数据，完整验收应先按“运行方法”准备数据。

```bash
cd src
python -m pytest -q
```

测试涵盖仿真器、动作掩码、HEFT、PPO、数据划分、搜索、混合策略、报告与实验流程。

尚未准备原始数据时，可以先在 `src/` 运行不依赖原始数据的仿真器与 PPO 掩码测试：

```bash
python -m pytest -q tests/test_simulator.py tests/test_ppo_mask.py
```

pytest 返回码为 `0` 表示本次选中的测试通过；完整验收还应检查输出中的跳过项，确认所需数据相关测试实际执行。一次完整演示也可用于运行检查：在项目根目录执行 `python run_demo.py --no-open`，确认新输出目录中 `status.json` 的 `status` 为 `complete`，并查看结果中的合法调度率。

### Linux 平台验收

仓库记录了两种 Linux 系统的虚拟机 CPU 验收：

| 平台 | Python | 已归档验收 |
| --- | --- | --- |
| openEuler 24.03 LTS-SP4 x86_64 | 3.11.6 | 188 项测试、2 个真实训练 episode、108 场景评测和新进程重载 |
| openKylin 2.0 SP2 x86_64 | 3.12.2 | 188 项测试、2 个真实训练 episode、108 场景评测和新进程重载 |

以上数量为归档时的记录，平台冒烟验收用于证明运行兼容性。环境与结果证据见 [openEuler 验收](tests/platform/openeuler_vm_20260930/README.md) 和 [openKylin 验收](tests/platform/openkylin_vm_20260930/README.md)。

在准备好依赖和数据的 Linux 环境中，可从 `src/` 运行新的验收：

```bash
python scripts/verify_platform.py --output outputs/platform_acceptance_new --execution-kind native
```

在虚拟机内执行时将 `native` 改为 `vm`。验收输出目录应尚不存在；脚本会生成环境信息、测试记录、训练及重载评测结果和 `acceptance.json`。该入口读取 Linux 系统信息，应在 Linux 环境执行。

## 已知限制

- **适用范围**：当前实现以离线 DAG 调度仿真为主，未提供真实云、边缘或终端集群的任务执行与部署接口。仿真成绩需在真实系统上进一步验证。
- **数据范围**：默认完整实验面向指定 Zenodo 数据的 50、100、300 任务规模；其他数据源和规模需要适配并重新验证。生成资源配置仍需要原始 DAG 数据。
- **资源分发**：完整原始数据与历史实验产物不随仓库分发；演示仅包含精选场景和权重。部分历史配置中的权重、搜索配置路径需改为本地实际路径。
- **评测范围**：归档成绩来自用于模型选择的验证集，6 场景演示也来自该验证集，均不能视为独立测试集上的泛化结论。
- **性能代价**：搜索和混合策略需要生成、校验多个完整调度，其推理耗时通常高于单一学习策略；完整 GPU 配置在 CPU 上可能运行较慢。
- **混合策略条件**：相对于候选方案的选择优势以各候选在同一确定性仿真器中成功完成并通过合法性检查为前提；该策略不证明全局最优，候选失败会报错。
- **模型兼容性**：评测配置的模型结构、观测设置与权重必须匹配。不同依赖、设备和随机种子可能影响数值及耗时，复现时应保留配置、划分与运行记录。
- **平台验证范围**：Linux 平台证据为所列版本的虚拟机 CPU 验收，不能自动覆盖其他系统版本、架构或 CUDA 环境；macOS 尚无归档验收记录。

## 目录结构

```text
.
├── README.md
├── run_demo.py / run_demo.cmd    # 根目录演示启动入口
├── demo/                        # 内置权重、场景、网页模板与演示结果
├── docs/                        # 项目说明书及相关文档
├── presentation/                # 展示材料目录
├── src/                         # 可安装的 Python 项目
│   ├── pyproject.toml           # 包信息、依赖与 pytest 配置
│   ├── configs/                 # 训练、评测、资源与平台配置
│   ├── data/                    # 数据说明、固定划分与本地原始数据
│   ├── scripts/                 # 数据准备、训练、评测、绘图与验收入口
│   ├── src/cpn_hrl_dag/         # 数据、环境、模型、策略、算法与评测实现
│   ├── tests/                   # Python 自动化测试
│   └── deploy/openeuler/        # openEuler 容器环境
└── tests/                       # 结果归档、演示审计与平台验收证据
```

项目说明书见 [docs/项目说明书.pdf](docs/项目说明书.pdf)。

## 常见问题

- **提示找不到 `cpn_hrl_dag`**：确认使用当前虚拟环境的 Python，并在项目根目录执行 `python -m pip install -e "./src[train,analysis,dev]"`。

- **提示缺少 Zenodo 数据**：完整训练需要手动准备原始压缩包；仅体验内置演示可以直接使用根目录的 `run_demo.py`。

- **出现 CUDA 不可用错误**：检查 PyTorch 和驱动，或复制配置并设置 `device: cpu`、新的 `output_dir`。完整训练在 CPU 上可能耗时较长。

- **演示资源哈希不匹配**：恢复与 `demo/manifest.json` 匹配的场景、权重和配置；主动更换资源时，应使用 `scripts/prepare_hybrid_demo.py` 重新生成演示包。

- **输出目录已存在**：为新运行指定空目录或新的路径，避免与已有结果混用。

## 演示视频

由于本项目演示视频过大，因此按照大赛要求采用网盘链接形式提供

具体链接：

通过网盘分享的文件：演示视频.mp4
链接: https://pan.baidu.com/s/1ekoLk1HXradGP4BRYFr8SQ 提取码: w59d 
--来自百度网盘超级会员v7的分享