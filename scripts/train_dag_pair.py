"""Distil HEFT-safe search trajectories into the joint DAG/resource graph policy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from cpn_hrl_dag.algorithms.dagger import CompletionRegretDAgger
from cpn_hrl_dag.algorithms.distillation import DAGPairDistiller, SearchTeacherCache
from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.datasets.split import SplitManager, SplitManifest
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator, summarize, write_report
from cpn_hrl_dag.models.dag_pair import DAGPairGraphActorCritic
from cpn_hrl_dag.policies.dag_pair import DAGPairGraphPolicy
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.utils.config import config_hash, load_config
from cpn_hrl_dag.utils.seed import seed_everything


def balanced_subset(scenarios: Iterable[Scenario], count: int) -> list[Scenario]:
    """Return a deterministic task-size-balanced subset."""

    values = list(scenarios)
    if count <= 0 or count >= len(values):
        return values
    groups: dict[int, list[Scenario]] = {}
    for scenario in values:
        groups.setdefault(scenario.num_tasks, []).append(scenario)
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


def build_search_policy(
    config: dict[str, Any],
    learned_policy: DAGPairGraphPolicy | None = None,
) -> HEFTSafePortfolioPolicy:
    """Construct the exact HEFT-safe deployment/search policy declared by YAML."""

    search = config["search"]
    return HEFTSafePortfolioPolicy(
        perturbation_candidates=int(search.get("perturbation_candidates", 16)),
        task_top_k=int(search.get("task_top_k", 2)),
        node_top_k=int(search.get("node_top_k", 2)),
        alternative_node_probability=float(search.get("alternative_node_probability", 0.15)),
        seed=int(config["experiment"]["seed"]),
        learned_policy=learned_policy,
        normalize_observations=True,
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


def build_teacher(config: dict[str, Any]) -> HEFTSafePortfolioPolicy:
    """Backward-compatible name for the fixed safe-search trajectory teacher."""

    return build_search_policy(config)


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` without silently accepting an unavailable CUDA request."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_model(observation: dict[str, Any], model_config: dict[str, Any]) -> DAGPairGraphActorCritic:
    """Build a variable-size graph model from feature dimensions, never task counts."""

    return DAGPairGraphActorCritic(
        observation["task_features"].shape[-1],
        observation["resource_features"].shape[-1],
        observation["pair_node_features"].shape[-1],
        observation["task_edge_features"].shape[-1],
        observation["resource_edge_features"].shape[-1],
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        heads=int(model_config.get("heads", 4)),
        task_layers=int(model_config.get("task_layers", 2)),
        resource_layers=int(model_config.get("resource_layers", 2)),
        dropout=float(model_config.get("dropout", 0.0)),
        heuristic_residual=bool(model_config.get("heuristic_residual", True)),
        residual_limit=float(model_config.get("residual_limit", 2.0)),
        heuristic_scale=float(model_config.get("heuristic_scale", 0.05)),
    )


def save_checkpoint(
    path: Path,
    model: DAGPairGraphActorCritic,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best_ratio: float,
    search_signature: str,
) -> None:
    """Persist all state needed to resume distillation or load inference."""

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "config_hash": config_hash(config),
            "epoch": int(epoch),
            "best_validation_ratio": float(best_ratio),
            "search_signature": search_signature,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, disable_cudnn=bool(config.get("device_options", {}).get("disable_cudnn", False)))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

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
    distillation = config["distillation"]
    method = str(distillation.get("method", "trajectory_bc")).lower()
    if method not in {"trajectory_bc", "regret_dagger"}:
        raise ValueError("distillation.method must be 'trajectory_bc' or 'regret_dagger'")
    training_scenarios = balanced_subset(splits["train"], int(distillation["train_scenarios"]))
    validation_scenarios = balanced_subset(
        splits["validation"], int(distillation["validation_scenarios"])
    )
    if {scenario.metadata["original_graph_id"] for scenario in training_scenarios} & {
        scenario.metadata["original_graph_id"] for scenario in validation_scenarios
    }:
        raise AssertionError("distillation train/validation topology leakage detected")

    search_signature = config_hash({"search": config["search"], "method": method})
    teachers = {}
    teacher_start = perf_counter()
    if method == "trajectory_bc":
        cache = SearchTeacherCache(output / "teacher_cache", search_signature)
        teacher_policy = build_teacher(config)
        for index, scenario in enumerate(training_scenarios, start=1):
            schedule = cache.get_or_compute(scenario, teacher_policy)
            teachers[scenario.scenario_id] = schedule
            print(
                f"teacher [{index:03d}/{len(training_scenarios):03d}] "
                f"tasks={scenario.num_tasks} ratio={schedule.ratio:.6f} "
                f"candidate={schedule.selected_candidate}",
                flush=True,
            )
        teacher_seconds = perf_counter() - teacher_start
        teacher_summary = {
            "method": method,
            "num_scenarios": len(teachers),
            "mean_ratio": float(np.mean([teacher.ratio for teacher in teachers.values()])),
            "max_ratio": float(np.max([teacher.ratio for teacher in teachers.values()])),
            "generation_seconds": teacher_seconds,
            "search_signature": search_signature,
        }
    else:
        teacher_seconds = 0.0
        teacher_summary = {
            "method": method,
            "num_scenarios": len(training_scenarios),
            "oracle": "model-visited legal actions + deterministic HEFT-rank/EFT completion",
            "generation_seconds": teacher_seconds,
            "search_signature": search_signature,
        }
    (output / "teacher_summary.json").write_text(
        json.dumps(teacher_summary, indent=2) + "\n", encoding="utf-8"
    )

    probe_env = CloudEdgeEndDAGEnv(normalize_observations=True)
    probe_env.reset(training_scenarios[0])
    probe = probe_env.get_flat_observation()
    model_config = config["model"]["dag_pair"]
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model(probe, model_config).to(device)
    if bool(distillation.get("freeze_graph_encoders", False)):
        for module in (model.task_encoder, model.resource_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("distillation configuration froze every model parameter")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(distillation.get("learning_rate", 3e-4)),
        weight_decay=float(distillation.get("weight_decay", 1e-5)),
    )
    if method == "trajectory_bc":
        trainer: DAGPairDistiller | CompletionRegretDAgger = DAGPairDistiller(
            model,
            optimizer,
            device,
            normalize_observations=True,
            states_per_scenario=int(distillation.get("states_per_scenario", 16)),
            value_weight=float(distillation.get("value_weight", 0.2)),
            max_grad_norm=float(distillation.get("max_grad_norm", 1.0)),
            disagreement_weight=float(distillation.get("disagreement_weight", 3.0)),
        )
    else:
        trainer = CompletionRegretDAgger(
            model,
            optimizer,
            device,
            normalize_observations=True,
            states_per_scenario=int(distillation.get("states_per_scenario", 8)),
            max_candidate_actions=int(distillation.get("max_candidate_actions", 6)),
            heuristic_task_candidates=int(distillation.get("heuristic_task_candidates", 2)),
            heuristic_node_candidates=int(distillation.get("heuristic_node_candidates", 2)),
            student_candidates=int(distillation.get("student_candidates", 2)),
            temperature=float(distillation.get("temperature", 0.025)),
            value_weight=float(distillation.get("value_weight", 0.1)),
            expected_regret_weight=float(distillation.get("expected_regret_weight", 1.0)),
            residual_weight=float(distillation.get("residual_weight", 0.01)),
            max_grad_norm=float(distillation.get("max_grad_norm", 1.0)),
        )
    evaluator = Evaluator({"normalize_observations": True})
    history: list[dict[str, Any]] = []
    training_start = perf_counter()

    validation_policy_name = str(distillation.get("validation_policy", "graph"))
    if validation_policy_name not in {"graph", "safe_search"}:
        raise ValueError("distillation.validation_policy must be 'graph' or 'safe_search'")

    def validation_policy(current_model: DAGPairGraphActorCritic) -> object:
        learned = DAGPairGraphPolicy(current_model, device, normalize_observations=True)
        return build_search_policy(config, learned) if validation_policy_name == "safe_search" else learned

    initial_records, initial_summary = evaluator.evaluate(
        validation_policy(model),
        validation_scenarios,
    )
    best_ratio = float(initial_summary["mean_ratio"])
    history.append(
        {
            "epoch": 0,
            "actor_loss": "",
            "value_loss": "",
            "action_accuracy": "",
            "validation_mean_ratio": best_ratio,
            "teacher_beta": "",
        }
    )
    save_checkpoint(output / "best.pt", model, optimizer, config, 0, best_ratio, search_signature)
    save_checkpoint(output / "latest.pt", model, optimizer, config, 0, best_ratio, search_signature)

    total_epochs = int(distillation["epochs"])
    patience = int(distillation.get("early_stop_patience", 0))
    minimum_delta = float(distillation.get("early_stop_min_delta", 1e-6))
    stale_epochs = 0
    for epoch in range(1, total_epochs + 1):
        if method == "trajectory_bc":
            assert isinstance(trainer, DAGPairDistiller)
            result = trainer.fit(training_scenarios, teachers, epochs=1, seed=seed + epoch)
            teacher_beta: float | str = ""
        else:
            assert isinstance(trainer, CompletionRegretDAgger)
            beta_start = float(distillation.get("teacher_beta_start", 0.75))
            beta_end = float(distillation.get("teacher_beta_end", 0.1))
            fraction = (epoch - 1) / max(total_epochs - 1, 1)
            teacher_beta = beta_start + fraction * (beta_end - beta_start)
            result = trainer.fit(
                training_scenarios,
                epochs=1,
                seed=seed + epoch,
                teacher_beta=float(teacher_beta),
            )
        records, summary = evaluator.evaluate(
            validation_policy(model),
            validation_scenarios,
        )
        ratio = float(summary["mean_ratio"])
        row = {
            "epoch": epoch,
            **asdict(result),
            "validation_mean_ratio": ratio,
            "teacher_beta": teacher_beta,
        }
        history.append(row)
        if ratio < best_ratio - minimum_delta:
            best_ratio = ratio
            save_checkpoint(output / "best.pt", model, optimizer, config, epoch, best_ratio, search_signature)
            stale_epochs = 0
        else:
            stale_epochs += 1
        save_checkpoint(output / "latest.pt", model, optimizer, config, epoch, best_ratio, search_signature)
        print(
            f"epoch={epoch} actor_loss={result.actor_loss:.6f} "
            f"accuracy={result.action_accuracy:.4f} validation_mean_ratio={ratio:.6f}",
            flush=True,
        )
        if patience > 0 and stale_epochs >= patience:
            print(f"early stopping after {stale_epochs} stale validation epochs", flush=True)
            break

    elapsed = perf_counter() - training_start
    with (output / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in history for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    # Load through a fresh model instance before the final smoke report.
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    loaded_model = build_model(probe, model_config).to(device)
    loaded_model.load_state_dict(checkpoint["model_state"])
    final_records, _ = evaluator.evaluate(
        validation_policy(loaded_model),
        validation_scenarios,
    )
    result_scope = str(config["experiment"].get("result_scope", "SMOKE TEST RESULT"))
    report_split = str(config["experiment"].get("report_split", "validation_smoke"))
    report = summarize(
        final_records,
        model=f"DAGPairGraph-{method}-{validation_policy_name}",
        split=report_split,
        seed=seed,
        config_hash=config_hash(config),
        training_time_seconds=elapsed,
        bootstrap_samples=int(config.get("evaluation", {}).get("bootstrap_samples", 0)),
    )
    report["result_scope"] = result_scope
    report["training_method"] = method
    report["validation_policy"] = validation_policy_name
    report["checkpoint_epoch"] = int(checkpoint["epoch"])
    report["best_validation_ratio"] = float(checkpoint["best_validation_ratio"])
    report["teacher_generation_seconds"] = teacher_seconds
    write_report(output / "validation_smoke", final_records, report)

    if bool(distillation.get("report_pure_policy", method == "regret_dagger")):
        pure_records, _ = evaluator.evaluate(
            DAGPairGraphPolicy(loaded_model, device, normalize_observations=True),
            validation_scenarios,
        )
        pure_report = summarize(
            pure_records,
            model=f"DAGPairGraph-{method}",
            split=report_split,
            seed=seed,
            config_hash=config_hash(config),
            training_time_seconds=elapsed,
            bootstrap_samples=int(config.get("evaluation", {}).get("bootstrap_samples", 0)),
        )
        pure_report["result_scope"] = result_scope
        pure_report["checkpoint_epoch"] = int(checkpoint["epoch"])
        write_report(output / "validation_pure", pure_records, pure_report)
    print(
        f"{result_scope} mean_ratio={report['mean_ratio']:.6f} "
        f"valid_schedule_rate={report['valid_schedule_rate']:.3f}; output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
