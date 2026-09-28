# Dataset audit

Audit date: 2026-09-10. Findings below come from the real files under
`data/raw`, not from dataset names or README assertions.

| Dataset | Version / provenance | Count | Task schema | Dependency schema | Resource / communication schema | Units, missing fields, adapter strategy | Licence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zenodo 18927122 STG JSON | Zenodo DOI `10.5281/zenodo.18927122`; CC-BY-4.0 | The supplied read-only `.tar.xz` release contains rnc50--rnc5000 homogeneous/heterogeneous families (180 variants per family), FPPPP controls, and packaged system configs. Default experiments select rnc50/100/300: 1,080 workflows and 540 base identities. | `tasks.<id>{cores,memory_required,features,data,duration,dependencies,tags}` | Child stores predecessor IDs; each incoming edge receives predecessor task `data` | packaged `system_configs.tar.xz`: `{tier,cores,memory,storage,features,processing_speed,data_transfer_rate}` | Duration is workload. The adapter reads archive members directly, preserves archive-member provenance, and never expands or modifies raw data. CPU/GPU device speeds are preserved by `HeterogeneousDeviceExecutionModel`. The compatibility network adapter uses min(endpoint maximum transfer rate) only to produce the framework's pairwise bandwidth matrix. | CC-BY-4.0 |

Every Scenario preserves `dataset_source`, `original_path`, `original_graph_id`,
`adapter_version`, original metadata, resource-generation seed, and resource
configuration ID. Split isolation is based on
`dataset_source:original_graph_id`; GrapheonRL homogeneous and heterogeneous
realizations of one STG are treated as the same topology family.

The official contest document in `doc/` takes priority if it later provides a
validation schema or evaluator. CPN-HRL is used only for the hierarchical
LSTM/GAT/PPO decomposition: its online CPN queue objective is not reused for
this offline DAG makespan problem.
