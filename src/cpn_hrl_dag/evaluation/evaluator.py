"""Evaluate any SchedulerPolicy against same-environment HEFT references."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.datasets.protocols import scenario_domain
from cpn_hrl_dag.policies.base import SchedulerPolicy
from cpn_hrl_dag.scenario.types import Scenario


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    scenario_id: str
    base_dag_id: str
    dataset_source: str
    num_tasks: int
    num_edges: int
    num_nodes: int
    num_cloud: int
    num_edge: int
    num_end: int
    domain: str
    policy: str
    makespan: float
    heft_makespan: float
    ratio: float
    valid_schedule: bool
    inference_time_ms: float


class Evaluator:
    """Evaluator that does not branch on policy implementation type."""

    def __init__(self, environment_kwargs: Mapping[str, Any] | None = None,
                 progress_callback: Callable | None = None) -> None:
        self.environment_kwargs = dict(environment_kwargs or {})
        self.progress_callback = progress_callback

    def evaluate(self, policy: SchedulerPolicy, scenarios: Iterable[Scenario]) -> tuple[list[EvaluationRecord], dict[str, float | int]]:
        records: list[EvaluationRecord] = []
        scenarios = list(scenarios)
        if self.progress_callback:
            self.progress_callback(0, len(scenarios), scenarios[0].scenario_id if scenarios else None, None)
        for scenario in scenarios:
            env = CloudEdgeEndDAGEnv(**self.environment_kwargs)
            high_observation, _ = env.reset(scenario)
            start = perf_counter()
            # Policy setup is part of inference.  This matters for planning
            # policies such as the HEFT-safe portfolio, whose reset computes
            # complete candidate schedules before replaying the winner.
            policy.reset(scenario)
            terminated = False
            info: dict[str, object] = {}
            while not terminated:
                ready = env.get_ready_mask()
                task = policy.select_task(high_observation, ready, deterministic=True)
                env.select_task(task)
                low_observation = env.get_low_observation(task)
                node = policy.select_node(low_observation, task, env.get_node_mask(task), deterministic=True)
                high_observation, _, terminated, truncated, info = env.step(task, node)
                if truncated:
                    raise RuntimeError("offline DAG environment cannot truncate episodes")
            elapsed = (perf_counter() - start) * 1000.0
            makespan, heft = float(info["final_makespan"]), float(info["heft_makespan"])
            tiers = [node.node_type.lower() for node in scenario.compute_nodes]
            domain = scenario_domain(scenario)
            records.append(EvaluationRecord(scenario.scenario_id, str(scenario.metadata["original_graph_id"]), scenario.dataset_source, scenario.num_tasks, len(scenario.dependencies), scenario.num_nodes, tiers.count("cloud"), tiers.count("edge"), tiers.count("end"), domain, policy.name, makespan, heft, makespan / heft, bool(info["valid_schedule"]), elapsed))
            if self.progress_callback:
                self.progress_callback(len(records), len(scenarios), scenario.scenario_id, makespan / heft)
        if not records:
            raise ValueError("evaluator requires at least one scenario")
        ratios = np.asarray([record.ratio for record in records])
        summary: dict[str, float | int] = {"num_scenarios": len(records), "mean_ratio": float(ratios.mean()), "std_ratio": float(ratios.std()), "median_ratio": float(np.median(ratios)), "p90_ratio": float(np.quantile(ratios, .9)), "valid_schedule_rate": float(np.mean([record.valid_schedule for record in records]))}
        return records, summary
