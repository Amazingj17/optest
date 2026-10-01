"""Regenerate aggregate reports from an evaluator-produced per-scene CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cpn_hrl_dag.evaluation import EvaluationRecord, summarize, write_report
from cpn_hrl_dag.utils.config import config_hash, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--config", help="YAML config used to recompute the canonical config hash")
    args = parser.parse_args()
    target = Path(args.evaluation_dir)
    source = target / "per_scene.csv"
    if not source.is_file():
        raise FileNotFoundError(f"missing evaluator export: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        records = []
        for row in csv.DictReader(handle):
            records.append(
                EvaluationRecord(
                    scenario_id=row["scenario_id"],
                    base_dag_id=row["base_dag_id"],
                    dataset_source=row["dataset_source"],
                    num_tasks=int(row["num_tasks"]),
                    num_edges=int(row["num_edges"]),
                    num_nodes=int(row["num_nodes"]),
                    num_cloud=int(row["num_cloud"]),
                    num_edge=int(row["num_edge"]),
                    num_end=int(row["num_end"]),
                    domain=row["domain"],
                    policy=row["policy"],
                    makespan=float(row["makespan"]),
                    heft_makespan=float(row["heft_makespan"]),
                    ratio=float(row["ratio"]),
                    valid_schedule=row["valid_schedule"].lower() == "true",
                    inference_time_ms=float(row["inference_time_ms"]),
                )
            )
    previous_path = target / "summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    summary = summarize(
        records,
        model=str(previous.get("model", records[0].policy)),
        split=str(previous.get("split", "unknown")),
        seed=int(previous.get("seed", 0)),
        config_hash=config_hash(load_config(args.config)) if args.config else str(previous.get("config_hash", "unknown")),
        training_time_seconds=float(previous.get("training_time_seconds", 0.0)),
        bootstrap_samples=args.bootstrap_samples,
    )
    write_report(target, records, summary)
    print(previous_path)


if __name__ == "__main__":
    main()
