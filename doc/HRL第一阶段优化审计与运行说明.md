# HRL 第一阶段优化：审计、诊断与运行说明

日期：2026-09-24。

本轮在已有第一阶段代码上修正实验语义和结果保存问题，提供可运行的冻结对照。
高层动态特征、搜索教师和采样温度优化尚未实施。没有启动 1728 episodes 的正式分支训练。
本轮诊断不构成纯 HRL 性能已经提升的证据。

## 1. 已修正的问题

| 位置 | 修正 |
| --- | --- |
| `algorithms/phases.py`、`hierarchical_trainer.py` | 恢复 `low_pretrain` 按固定 HEFT rank 选任务的历史语义；冻结高层默认确定性执行，与部署策略使用相同的精度和并列处理 |
| `algorithms/ppo.py` | 冻结时清除旧梯度；诊断关闭时不计算动作统计；零回报方差时 explained variance 记录 null |
| `evaluation/diagnostics.py` | 区分偏离模型 argmax 与偏离最小 EFT；补充低层单/多合法动作统计；固定规则缺少网络分布不再误报采样异常 |
| `evaluation/diagnostics.py` | CSV 按 episode 增量追加，避免每次重写全部历史；关闭时写 `diagnostics_by_phase.json`，明确采用 episode 均值而非合并分位数 |
| `scripts/train_main_comparison.py` | 分叉重新评估实际加载的权重，并立即保存初始 best；后续全部退化时仍保留起始模型 |
| 分叉初始化 | 校验配置哈希、来源 step、模型/奖励/数据/资源/seed；明确 optimizer 继承或重置，fresh scheduler 从配置学习率开始 |
| 预算统计 | 实际 `optimizer_updates` 统计 minibatch 的 `optimizer.step`；`ppo_update_calls` 单独记录 episode 级 update 调用 |
| 阶段调度 | 按逐 episode 的阶段对象执行，同名阶段可有不同冻结执行模式，不再被字典覆盖 |
| 入口保护 | 分叉配置缺少 `--fork-from` 时拒绝；正式入口拒绝 proxy 和非确定性验证；preflight 移除删除旧输出的选项 |
| 只读诊断 | 加入“高层贪心＋低层采样”和“高层贪心＋固定 EFT”模式；更正把高低层共同变化误称为高层效果的字段 |

旧权重张量结构不变。`high_train` 仍映射到 `high_only_eft`。
固定 EFT 使用环境的 float64 精确值；对极接近并列的 EFT，相比早期 float32 特征排序可能存在微小差别。
开关诊断的动作轨迹、更新后参数和 RNG 一致性有测试；不声称所有历史版本训练逐位一致。

修改前源码快照：`outputs/code_backups/hrl_stage1_20260924_184757/`。
原始 `outputs/gpu/outputs/all_models_seed2026/` 结果与权重未被写入。

## 2. 当前训练样例诊断

来源：第一轮 `best.pt`，`global_step=864`。
SHA256：`db0aef70a80943d5b0ffd12d7babfbdb71a7e0e1826034e2eb0447970bec6c40`。

按任务规模 50/100/300 × 同构/异构，固定 seed=2026，从训练集每组选择一个场景，共 6 个。
随机模式每场景执行 3 次；确定性模式执行 1 次；CPU 单线程。

| 执行方式 | 平均 makespan/HEFT |
| --- | ---: |
| 高层贪心＋低层贪心 | 1.001698 |
| 高层采样＋低层采样 | 1.189064 |
| 高层采样＋固定最小 EFT | 1.022211 |
| 高层贪心＋低层采样 | 1.160016 |
| 高层贪心＋固定最小 EFT | 1.001698 |

所有诊断调度合法。结果保存在：
`outputs/hrl_stage1_audit_20260924/best_diagnostic/`。

这说明第一轮模型的低层采样与贪心行为存在明显差距，支持优先检查低层探索。
它不证明后期训练退化的因果关系，也不能替代完整 108 validation 的结果。
相同 seed 不意味着两种模式访问同一状态、抽到同一动作；不能把模式均值差直接当作因果贡献。

## 3. 分叉语义与字段

- A：`high_only_eft`，高层采样和更新，低层固定最小 EFT。
- B：`low_only_frozen_high`，高层冻结并确定性执行，低层采样和更新。
- `stochastic` 高层冻结模式仍可显式配置，但不是默认 B。
- 两者从同一 step=864 checkpoint 初始化，各跑 1728 episodes，即训练场景再完整覆盖两轮。
- 两分支从 seed=2026 重新生成相同场景排列。这不是恢复原训练的第二、三轮排列，也不是 exact resume。
- 场景预算相同不意味着计算量相同；策略访问状态也可能不同。

| 字段 | 默认/推荐值 | 说明 |
| --- | --- | --- |
| `training.diagnostics` | false；对照配置 true | 启用增量诊断 |
| `training.phases` | 缺省 | 缺省按旧预算；显式配置时以阶段表为准 |
| `frozen_high_mode` | deterministic | 只用于 `low_only_frozen_high` |
| `fork.optimizer_restore` | restore | 继承 Adam 状态；reset 则保留新优化器 |
| `fork.scheduler_restore` | fresh | 重建 scheduler，起始学习率取当前模型配置；restore 才继承源 scheduler |
| `fork.require_checkpoint` | 生成的分叉配置 true | 防止漏写来源后从零训练 |
| `fork.expected_source_step` | 正式对照 864 | 拒绝误用最终 latest.pt |

显式阶段表的学习率衰减预算按各层实际参与更新的 episode 数计算；冻结期间 scheduler 不推进。
旧配置保留原 scheduler 总预算。
重置优化器时不允许单独继承旧 scheduler。

`phase_plan.json` 的预计更新预算仍保留兼容字段 `optimizer_updates`，其中
`update_count_unit` 明确为 **PPO update 调用数**；真正完成的 `budget.json/executed/optimizer_updates`
单位为 **optimizer.step minibatch 数**。比较时必须同时读取单位字段。

## 4. 运行命令

在项目根目录执行。下面的命令可用于 PowerShell 或常见 Linux shell，每条均为单行。

### 生成新配置

```shell
python scripts/prepare_stage1_forks.py --source-checkpoint outputs/gpu/outputs/all_models_seed2026/training/residual_hrl/best.pt --output outputs/hrl_stage1_next --episodes 1728 --coverage-epochs 2 --diagnostics
```

默认以 checkpoint 内的配置为模板，避免继承历史 fast 配置的不同模型参数。
输出目录必须是新目录。跨机器生成时用 `--output-root` 指定训练机路径；数据路径仍需在训练机有效。

### 只读预检（不执行 probe 更新）

```shell
python scripts/preflight_stage1_run.py --config outputs/hrl_stage1_next/configs/high_only_eft.yaml --fork-from outputs/gpu/outputs/all_models_seed2026/training/residual_hrl/best.pt --skip-probe
python scripts/preflight_stage1_run.py --config outputs/hrl_stage1_next/configs/low_only_frozen_high.yaml --fork-from outputs/gpu/outputs/all_models_seed2026/training/residual_hrl/best.pt --skip-probe
```

不加 `--skip-probe` 会在临时内存模型上执行一次训练 episode，不写 checkpoint、不启动正式训练。
preflight 不再支持删除输出目录，目录冲突必须换新名称。

### 正式对照命令（本轮未执行）

```shell
python scripts/train_main_comparison.py --config outputs/hrl_stage1_next/configs/high_only_eft.yaml --fork-from outputs/gpu/outputs/all_models_seed2026/training/residual_hrl/best.pt
python scripts/train_main_comparison.py --config outputs/hrl_stage1_next/configs/low_only_frozen_high.yaml --fork-from outputs/gpu/outputs/all_models_seed2026/training/residual_hrl/best.pt
```

本轮已经生成的等价配置位于 `outputs/hrl_stage1_audit_20260924/forks/configs/`，可直接替换上面的配置路径。
正式验证仍为原 108 场景，禁用 proxy，不评估 test。

### 重做只读样例诊断

```shell
python scripts/diagnose_hrl_execution.py --checkpoint outputs/gpu/outputs/all_models_seed2026/training/residual_hrl/best.pt --output outputs/hrl_stage1_diagnostic_new --threads 1 --repeats 3
```

## 5. 验证与下一步

回归测试涵盖诊断不改变轨迹/参数/RNG、冻结已有 optimizer 状态、历史低层预训练规则、
checkpoint 分叉校验、fresh scheduler 起始学习率、初始 best 保留、真实 optimizer.step 计数、
两分支场景顺序一致，以及写入保护。

```shell
python -m pytest tests/test_stage1_diagnostics.py tests/test_hrl_stage1_regressions.py tests/test_ppo_mask.py tests/test_ppo_mode.py tests/test_server_comparison.py tests/test_training_utilities.py -q
```

真实 checkpoint smoke 配置：`outputs/hrl_stage1_audit_20260924/real_checkpoint_smoke.yaml`。
仅执行 2 个低层训练 episodes，随后全量验证；它只证明流程可执行，不是性能实验。
其最终状态、实际预算和验证历史以同目录下 `real_checkpoint_smoke/` 的产物为准。

本轮实测已完成：

- 58 项相关回归测试通过（完整输出：`outputs/hrl_stage1_audit_20260924/pytest_results.txt`）。
- 真实 checkpoint smoke：2 episodes、400 decisions、8 次低层 optimizer.step；高层 0 次更新，低层 scheduler 推进 2 次。
- 起始权重全量验证 ratio 为 0.9995307035；两次低层更新后为 1.0014459517，出现退化。
- 选中的 best 正确保留起始 step=864，重新验证仍为 0.9995307035，合法率 100%。
- 2 个训练场景并不构成 1 次完整覆盖，`coverage_epochs=2/864`，`complete_coverage_epochs=0`。

以上不说明低层永远不能学习，只说明此次 smoke 没有改善，且保存与计数行为符合预期。

后续先运行 A/B 对照。如果 A 稳定、B 退化，再针对低层探索引入经过验证的约束或温度调度；
任何分布改变必须在动作采样、存储 log_prob 与 PPO 更新中保持一致。
如果两者都难以改善，再单独引入高层动态资源特征和训练集搜索教师。
不要同时改变奖励、网络和探索参数后宣称定位了原因。
