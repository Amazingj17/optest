# CPN-HRL-DAG 提交目录

本目录为代码提交副本，原项目代码及文档保持原位。src/ 是可独立安装的工程根目录，其内部 src/cpn_hrl_dag/ 为 Python 源码包。保留原有相对路径，未修改算法、配置或测试代码。

## 目录

- src/：源代码、运行脚本、配置、部署文件、数据划分清单及原始测试代码。
- tests/：测试入口和 SHA256 复制校验清单；实际测试代码位于 src/tests/。
- docs/：预留项目说明书.pdf、项目创新说明.pdf、成员分工及主要贡献说明.pdf、人工智能及第三方工具使用说明.pdf。
- demo/：预留与赛道相适配的可验证交付物。
- presentation/：预留决赛现场演示 PPT 和决赛演示视频。
- 根目录：预留作品原创承诺书.pdf。

文档、PPT、视频未复制或移动，未创建空白 PDF 冒充正式材料。此目录目前是代码整理副本，不是材料齐全的最终提交包。

## 安装和测试

从本提交目录运行：

```powershell
cd src
python -m pip install -e ".[train,analysis,dev]"
python -m pytest tests
```

也可在提交目录运行 `powershell -ExecutionPolicy Bypass -File tests/run_tests.ps1`。

训练、评估脚本须在 src/ 工程根目录运行。原始数据、模型权重和历史实验 outputs/ 未打包；涉及这些资源的运行需另行准备。数据来源为 https://zenodo.org/records/18927122 ，按配置放置到 src/data/raw/，权重按对应配置放置到 src/outputs/。现有完整操作文档仍在原项目 README.md、doc/、docs/ 中。

本次验证了复制文件与原文件的 SHA256 一致性，没有运行训练或完整测试。文件对应关系见 tests/copy_manifest.csv。

## 演示方案对比

打开 `demo/index.html`，查看 HEFT、GraphPPO、TierMAPPO 和 HRL 组合搜索的真实调度结果。包含 50、100、300 个任务的同构与异构场景，共六个样例。点击方案可查看对应时间线与任务分配。

模型权重和场景均已打包，运行 `python run_demo.py` 可重新计算。GraphPPO 与 TierMAPPO 使用与 HRL 同批次、同 seed 2026 的已训练模型；组合搜索保持原有算法。
