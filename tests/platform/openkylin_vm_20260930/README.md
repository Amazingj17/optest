# openKylin 2.0 SP2 虚拟机 CPU 验收

系统：openKylin 2.0 SP2，x86_64，Python 3.12.2。
在虚拟机中完成 188 项测试、2 个训练 episode、模型保存与新进程重载评测。
验证集：108 个场景、54 个基础 DAG；HRL 与 HEFT 合法调度率均为 1.0。
HRL mean_ratio = 0.9996353632803595；HEFT mean_ratio = 1.0。
这是平台运行验收，不作为充分训练后的正式性能成绩。
配置、依赖版本、数据校验值和逐场景结果见本目录。
原始 Zenodo 数据与模型权重保留在本地，不纳入 Git 仓库。
