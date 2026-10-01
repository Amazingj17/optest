# Archived three-seed result summary

These files are unchanged copies of the lightweight report, CSV tables and
verification JSON from the local batch:

`outputs/gpu/outputs/multiseed_20260925_122243/multiseed_summary/`

Start with [the verified report](三种子训练验收与结果报告.md) or
[the method summary](mean_std.csv). The report was checked on 2026-09-28.

This is a result snapshot, not the full experiment archive. Checkpoints,
per-scenario schedules, training logs, runtime records and the original
`build_report.py` are retained locally under `outputs/` and are not included
in Git. Paths to those artifacts in the report describe the original run.

Three seeds share the same 108 validation scenarios / 54 base DAGs. Cross-seed
standard deviations use ddof=1. Validation was used for checkpoint selection;
these are not independent test results. Search randomness also changes with
the experiment seed. Most of the hybrid gain comes from search.
