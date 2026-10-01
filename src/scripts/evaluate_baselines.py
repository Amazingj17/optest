"""Evaluate all non-learning baselines after loading a split only once."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.datasets.split import SplitManager, SplitManifest
from cpn_hrl_dag.evaluation import Evaluator, summarize, write_report
from cpn_hrl_dag.policies.heuristics import GreedyEFTPolicy, HEFTPolicy, RandomPolicy
from cpn_hrl_dag.utils.config import config_hash, load_config
from cpn_hrl_dag.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, disable_cudnn=bool(config.get("device_options", {}).get("disable_cudnn", False)))
    dataset = config["dataset"]
    scenarios = load_scenarios(
        default_registry(dataset.get("grapheonrl_system_configs"), dataset.get("archive_task_counts")),
        dataset["roots"],
        limit_per_dataset=dataset.get("limit_per_dataset"),
    )
    manifest = SplitManifest.read(dataset["split_manifest"])
    splits = materialize_resources(
        SplitManager.apply(scenarios, manifest),
        config.get("resources"),
        manifest,
        seed,
    )
    selected = splits[args.split]
    evaluator = Evaluator(
        {
            "include_heft_features": bool(config.get("model", {}).get("high", {}).get("include_heft_features", True)),
            "normalize_observations": bool(config.get("environment", {}).get("normalize_observations", False)),
        }
    )
    policies = (HEFTPolicy(), GreedyEFTPolicy(), RandomPolicy(seed))
    for policy in policies:
        records, _ = evaluator.evaluate(policy, selected)
        report = summarize(
            records,
            model=policy.name,
            split=args.split,
            seed=seed,
            config_hash=config_hash(config),
            bootstrap_samples=int(config.get("evaluation", {}).get("bootstrap_samples", 0)),
        )
        output = Path(config["output_dir"]) / f"eval_{policy.name}_{args.split}"
        write_report(output, records, report)
        print(f"{policy.name}: mean_ratio={report['mean_ratio']:.6f}; output={output}")


if __name__ == "__main__":
    main()
