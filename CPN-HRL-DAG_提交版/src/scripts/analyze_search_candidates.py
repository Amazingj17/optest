"""Audit every safe-search candidate on a fixed, size-balanced split subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.datasets.split import SplitManager, SplitManifest
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.utils.config import load_config
from cpn_hrl_dag.utils.seed import seed_everything


def balanced_subset(scenarios: Iterable[Scenario], count: int) -> list[Scenario]:
    """Select deterministic, approximately size-balanced scenarios."""

    values = list(scenarios)
    if count <= 0 or count >= len(values):
        return values
    groups: dict[int, list[Scenario]] = defaultdict(list)
    for scenario in values:
        groups[scenario.num_tasks].append(scenario)
    chosen: list[Scenario] = []
    per_group = max(1, count // len(groups))
    for task_count in sorted(groups):
        group = sorted(groups[task_count], key=lambda item: item.scenario_id)
        indices = np.linspace(0, len(group) - 1, min(per_group, len(group)), dtype=int)
        chosen.extend(group[int(index)] for index in indices)
    chosen_ids = {scenario.scenario_id for scenario in chosen}
    for scenario in sorted(values, key=lambda item: item.scenario_id):
        if len(chosen) >= count:
            break
        if scenario.scenario_id not in chosen_ids:
            chosen.append(scenario)
            chosen_ids.add(scenario.scenario_id)
    return chosen[:count]


def candidate_family(name: str) -> str:
    """Collapse depth/seed-specific candidate names for aggregate reporting."""

    if name.startswith("beam_"):
        return "beam_search"
    if name.startswith("block_"):
        return "critical_block"
    if name.startswith("perturbed_heft_"):
        return "perturbed_heft"
    if name.startswith("lds_task_"):
        return "lds_task"
    if name.startswith("lds_node_"):
        return "lds_node"
    if name.startswith("local_"):
        return "local_greedy" if name.endswith("_greedy") else "local_preserve"
    return name


def build_policy(config: dict[str, Any]) -> HEFTSafePortfolioPolicy:
    """Construct the configured deterministic safe-search policy."""

    search = config.get("search", config.get("portfolio", {}))
    return HEFTSafePortfolioPolicy(
        perturbation_candidates=int(search.get("perturbation_candidates", 0)),
        task_top_k=int(search.get("task_top_k", 2)),
        node_top_k=int(search.get("node_top_k", 2)),
        alternative_node_probability=float(search.get("alternative_node_probability", 0.15)),
        seed=int(config["experiment"]["seed"]),
        include_peft=bool(search.get("include_peft", False)),
        peft_lookahead_weights=tuple(search.get("peft_lookahead_weights", [1.0])),
        include_cpop=bool(search.get("include_cpop", False)),
        heft_communication_scales=tuple(search.get("heft_communication_scales", [])),
        task_discrepancy_candidates=int(search.get("task_discrepancy_candidates", 0)),
        node_discrepancy_candidates=int(search.get("node_discrepancy_candidates", 0)),
        local_search_rounds=int(search.get("local_search_rounds", 0)),
        local_search_critical_tasks=int(search.get("local_search_critical_tasks", 0)),
        local_search_node_alternatives=int(search.get("local_search_node_alternatives", 1)),
        local_search_task_alternatives=int(search.get("local_search_task_alternatives", 0)),
        local_search_beam_width=int(search.get("local_search_beam_width", 1)),
        local_search_prefix_cache=bool(search.get("local_search_prefix_cache", True)),
        local_search_result_cache=bool(search.get("local_search_result_cache", False)),
        local_search_block_rounds=int(search.get("local_search_block_rounds", 0)),
        local_search_blocks=int(search.get("local_search_blocks", 4)),
        local_search_block_max_size=int(search.get("local_search_block_max_size", 3)),
        local_search_block_node_alternatives=int(search.get("local_search_block_node_alternatives", 2)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=18, help="Size-balanced scenario count; <=0 means all")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    dataset = config["dataset"]
    scenarios = load_scenarios(
        default_registry(dataset.get("grapheonrl_system_configs"), dataset.get("archive_task_counts")),
        dataset["roots"],
        limit_per_dataset=dataset.get("limit_per_dataset"),
    )
    manifest = SplitManifest.read(dataset["split_manifest"])
    split = materialize_resources(
        SplitManager.apply(scenarios, manifest),
        config.get("resources"),
        manifest,
        seed,
    )[args.split]
    selected = balanced_subset(split, args.limit)
    policy = build_policy(config)

    rows: list[dict[str, Any]] = []
    winners: Counter[str] = Counter()
    portfolio_ratios: list[float] = []
    start = perf_counter()
    for index, scenario in enumerate(selected, start=1):
        scenario_start = perf_counter()
        policy.reset(scenario)
        elapsed_ms = (perf_counter() - scenario_start) * 1000.0
        heft_makespan = float(policy.candidates[0].result.makespan)
        ratio = float(policy.selected_makespan / heft_makespan)
        if ratio > 1.0 + 1e-12:
            raise AssertionError("safe search returned a result worse than canonical HEFT")
        portfolio_ratios.append(ratio)
        winners[candidate_family(policy.selected_candidate)] += 1
        for candidate in policy.candidates:
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "num_tasks": scenario.num_tasks,
                    "candidate": candidate.name,
                    "family": candidate_family(candidate.name),
                    "makespan": candidate.result.makespan,
                    "heft_makespan": heft_makespan,
                    "ratio": candidate.result.makespan / heft_makespan,
                    "selected": candidate.name == policy.selected_candidate,
                    "scenario_search_time_ms": elapsed_ms,
                }
            )
        print(
            f"[{index:03d}/{len(selected):03d}] tasks={scenario.num_tasks:3d} "
            f"ratio={ratio:.6f} winner={policy.selected_candidate} time_ms={elapsed_ms:.1f}",
            flush=True,
        )

    elapsed = perf_counter() - start
    family_ratios: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family_ratios[str(row["family"])].append(float(row["ratio"]))
    summary = {
        "split": args.split,
        "num_scenarios": len(selected),
        "mean_ratio": float(np.mean(portfolio_ratios)),
        "std_ratio": float(np.std(portfolio_ratios)),
        "max_ratio": float(np.max(portfolio_ratios)),
        "mean_search_time_ms": elapsed * 1000.0 / len(selected),
        "winner_counts": dict(sorted(winners.items())),
        "candidate_family_mean_ratio": {
            family: float(np.mean(ratios)) for family, ratios in sorted(family_ratios.items())
        },
    }
    output = Path(args.output) if args.output else Path(config["output_dir"]) / f"search_analysis_{args.split}_{len(selected)}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
