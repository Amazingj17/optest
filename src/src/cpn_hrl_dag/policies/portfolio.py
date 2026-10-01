"""HEFT-safe deterministic schedule portfolio.

The policy plans complete schedules in :meth:`reset` and replays the best one
through the normal two-stage policy interface.  Standard HEFT is always a
candidate, so candidate search can improve a schedule but can never make its
makespan worse than the project-owned HEFT implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cpn_hrl_dag.baselines.advanced import CPOPScheduler, PEFTScheduler, ScaledHEFTScheduler
from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.communication_model import MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import execution_model_for_scenario
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator, SimulationResult

from .base import SchedulerPolicy


@dataclass(frozen=True, slots=True)
class PortfolioCandidate:
    """Auditable result of one complete candidate scheduler."""

    name: str
    result: SimulationResult


@dataclass(frozen=True, slots=True)
class _DiscrepancyPoint:
    """One stable depth at which to try the second-best local decision."""

    depth: int
    kind: str
    score: float


def _stable_seed(seed: int, scenario_id: str, candidate: int) -> int:
    payload = f"{seed}:{scenario_id}:{candidate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class HEFTSafePortfolioPolicy(SchedulerPolicy):
    """Select the best of HEFT, optional learned policy, and perturbed HEFT.

    Perturbations only change choices within the top-k ready tasks and EFT
    nodes.  All candidates use :class:`ScheduleSimulator`, and deterministic
    tie-breaking always favours canonical HEFT.
    """

    name = "heft_safe_portfolio"

    def __init__(
        self,
        *,
        perturbation_candidates: int = 16,
        task_top_k: int = 2,
        node_top_k: int = 2,
        alternative_node_probability: float = 0.15,
        seed: int = 0,
        learned_policy: SchedulerPolicy | None = None,
        normalize_observations: bool = False,
        include_peft: bool = False,
        peft_lookahead_weights: Sequence[float] = (1.0,),
        include_cpop: bool = False,
        heft_communication_scales: Sequence[float] = (),
        task_discrepancy_candidates: int = 0,
        node_discrepancy_candidates: int = 0,
        local_search_rounds: int = 0,
        local_search_critical_tasks: int = 0,
        local_search_node_alternatives: int = 1,
        local_search_task_alternatives: int = 0,
        local_search_beam_width: int = 1,
        local_search_prefix_cache: bool = True,
        local_search_result_cache: bool = False,
        local_search_block_rounds: int = 0,
        local_search_blocks: int = 4,
        local_search_block_max_size: int = 3,
        local_search_block_node_alternatives: int = 2,
    ) -> None:
        if perturbation_candidates < 0:
            raise ValueError("perturbation_candidates must be non-negative")
        if task_top_k <= 0 or node_top_k <= 0:
            raise ValueError("portfolio top-k values must be positive")
        if not 0.0 <= alternative_node_probability <= 1.0:
            raise ValueError("alternative_node_probability must be in [0, 1]")
        if task_discrepancy_candidates < 0 or node_discrepancy_candidates < 0:
            raise ValueError("discrepancy candidate counts must be non-negative")
        if local_search_rounds < 0 or local_search_critical_tasks < 0:
            raise ValueError("local-search rounds and task count must be non-negative")
        if local_search_node_alternatives <= 0:
            raise ValueError("local_search_node_alternatives must be positive")
        if local_search_task_alternatives < 0:
            raise ValueError("local_search_task_alternatives must be non-negative")
        if local_search_beam_width <= 0:
            raise ValueError("local_search_beam_width must be positive")
        if local_search_block_rounds < 0:
            raise ValueError("local_search_block_rounds must be non-negative")
        if local_search_blocks <= 0 or local_search_block_node_alternatives <= 0:
            raise ValueError("local-search block counts must be positive")
        if local_search_block_max_size < 2:
            raise ValueError("local_search_block_max_size must be at least two")
        peft_weights = tuple(float(value) for value in peft_lookahead_weights)
        communication_scales = tuple(float(value) for value in heft_communication_scales)
        if include_peft and not peft_weights:
            raise ValueError("include_peft requires at least one lookahead weight")
        if any(not np.isfinite(value) or value < 0.0 for value in peft_weights):
            raise ValueError("PEFT lookahead weights must be finite and non-negative")
        if any(not np.isfinite(value) or value < 0.0 for value in communication_scales):
            raise ValueError("HEFT communication scales must be finite and non-negative")
        self.perturbation_candidates = int(perturbation_candidates)
        self.task_top_k = int(task_top_k)
        self.node_top_k = int(node_top_k)
        self.alternative_node_probability = float(alternative_node_probability)
        self.seed = int(seed)
        self.learned_policy = learned_policy
        self.normalize_observations = bool(normalize_observations)
        self.include_peft = bool(include_peft)
        self.peft_lookahead_weights = peft_weights
        self.include_cpop = bool(include_cpop)
        self.heft_communication_scales = communication_scales
        self.task_discrepancy_candidates = int(task_discrepancy_candidates)
        self.node_discrepancy_candidates = int(node_discrepancy_candidates)
        self.local_search_rounds = int(local_search_rounds)
        self.local_search_critical_tasks = int(local_search_critical_tasks)
        self.local_search_node_alternatives = int(local_search_node_alternatives)
        self.local_search_task_alternatives = int(local_search_task_alternatives)
        self.local_search_beam_width = int(local_search_beam_width)
        self.local_search_prefix_cache = bool(local_search_prefix_cache)
        self._prefix_incumbent: SimulationResult | None = None
        self._prefix_states: dict[int, ScheduleSimulator] = {}
        self.local_search_result_cache = bool(local_search_result_cache)
        self._replay_results: dict[tuple[tuple[int, int], ...], SimulationResult] = {}
        self.replay_cache_hits = 0
        self.replay_cache_skipped_tasks = 0
        self.local_search_block_rounds = int(local_search_block_rounds)
        self.local_search_blocks = int(local_search_blocks)
        self.local_search_block_max_size = int(local_search_block_max_size)
        self.local_search_block_node_alternatives = int(local_search_block_node_alternatives)
        advanced_search = (
            self.include_peft
            or self.include_cpop
            or bool(self.heft_communication_scales)
            or self.task_discrepancy_candidates > 0
            or self.node_discrepancy_candidates > 0
            or self.local_search_rounds > 0
            or self.local_search_block_rounds > 0
        )
        if learned_policy is not None:
            if self.local_search_rounds > 0 or self.local_search_block_rounds > 0:
                self.name = "cpn_hrl_dag_lns_safe"
            else:
                self.name = "cpn_hrl_dag_search_safe" if advanced_search else "cpn_hrl_dag_heft_safe"
        elif self.local_search_rounds > 0 or self.local_search_block_rounds > 0:
            self.name = "heft_safe_lns"
        else:
            self.name = "heft_safe_search" if advanced_search else "heft_safe_portfolio"
        self.candidates: tuple[PortfolioCandidate, ...] = ()
        self.selected_candidate = ""
        self.selected_makespan = float("inf")
        self._decisions: tuple[tuple[int, int], ...] = ()
        self._cursor = 0

    @property
    def planned_decisions(self) -> tuple[tuple[int, int], ...]:
        """Return the immutable winning trajectory for audit/distillation."""

        if not self._decisions:
            raise RuntimeError("portfolio must be reset before reading its planned decisions")
        return self._decisions

    def reset(self, scenario: Scenario) -> None:
        self._prefix_incumbent = None
        self._prefix_states.clear()
        self._replay_results.clear()
        self.replay_cache_hits = 0
        self.replay_cache_skipped_tasks = 0
        execution = execution_model_for_scenario(scenario.dataset_source)
        communication = MatrixCommunicationModel()
        heft_result, analysis = HEFTScheduler(execution, communication).schedule(scenario)
        candidates = [PortfolioCandidate("heft", heft_result)]
        if self.include_cpop:
            cpop_result, _ = CPOPScheduler(execution, communication).schedule(scenario)
            candidates.append(PortfolioCandidate("cpop", cpop_result))
        if self.include_peft:
            for weight in self.peft_lookahead_weights:
                peft_result, _ = PEFTScheduler(
                    execution,
                    communication,
                    lookahead_weight=weight,
                ).schedule(scenario)
                candidates.append(PortfolioCandidate(f"peft_w{weight:g}", peft_result))
        for scale in self.heft_communication_scales:
            if abs(scale - 1.0) <= 1e-12:
                continue
            scaled_result, _ = ScaledHEFTScheduler(
                execution,
                communication,
                communication_scale=scale,
            ).schedule(scenario)
            candidates.append(PortfolioCandidate(f"heft_comm_x{scale:g}", scaled_result))
        if self.learned_policy is not None:
            learned = self._rollout_policy(scenario, self.learned_policy, self.normalize_observations)
            candidates.append(PortfolioCandidate(self.learned_policy.name, learned))
        discrepancy_points = self._find_discrepancy_points(
            scenario,
            analysis.upward_rank,
            heft_result.makespan,
            execution,
            communication,
        )
        for point in discrepancy_points:
            result = self._single_discrepancy(
                scenario,
                analysis.upward_rank,
                point,
                execution,
                communication,
            )
            candidates.append(PortfolioCandidate(f"lds_{point.kind}_{point.depth:03d}", result))
        for index in range(self.perturbation_candidates):
            result = self._perturbed_heft(
                scenario,
                analysis.upward_rank,
                index,
                execution,
                communication,
            )
            candidates.append(PortfolioCandidate(f"perturbed_heft_{index:03d}", result))
        for candidate in candidates:
            self._validate_candidate(candidate)
            self._remember_replay(candidate.result)

        incumbent = min(
            enumerate(candidates),
            key=lambda item: (float(item[1].result.makespan), item[0]),
        )[1]
        for round_index in range(self.local_search_rounds if self.local_search_beam_width == 1 else 0):
            moves = self._critical_path_node_moves(
                scenario,
                incumbent,
                round_index,
                execution,
                communication,
            )
            moves.extend(
                self._critical_path_task_moves(
                    scenario,
                    incumbent,
                    analysis.upward_rank,
                    round_index,
                    execution,
                    communication,
                )
            )
            if not moves:
                break
            for candidate in moves:
                self._validate_candidate(candidate)
            candidates.extend(moves)
            next_incumbent = min(
                enumerate(candidates),
                key=lambda item: (float(item[1].result.makespan), item[0]),
            )[1]
            if next_incumbent.result.makespan >= incumbent.result.makespan - 1e-12:
                break
            incumbent = next_incumbent
        if self.local_search_beam_width > 1:
            self._beam_search(scenario, candidates, analysis.upward_rank, execution, communication)
        for round_index in range(self.local_search_block_rounds):
            incumbent = min(candidates, key=lambda candidate: float(candidate.result.makespan))
            moves = self._critical_block_moves(scenario, incumbent, round_index, execution, communication)
            for candidate in moves:
                self._validate_candidate(candidate)
            candidates.extend(moves)
            if not moves or min(candidate.result.makespan for candidate in moves) >= incumbent.result.makespan - 1e-12:
                break
        best_index, best = min(
            enumerate(candidates),
            key=lambda item: (float(item[1].result.makespan), item[0]),
        )
        del best_index
        if best.result.makespan > heft_result.makespan + 1e-9:
            raise AssertionError("HEFT-safe portfolio selected a schedule worse than HEFT")
        by_task = {entry.task_position: entry.node_position for entry in best.result.entries}
        self._decisions = tuple((task, by_task[task]) for task in best.result.decision_order)
        self._cursor = 0
        self.candidates = tuple(candidates)
        self.selected_candidate = best.name
        self.selected_makespan = float(best.result.makespan)
        self._prefix_incumbent = None
        self._prefix_states.clear()
        self._replay_results.clear()

    @staticmethod
    def _schedule_key(candidate: PortfolioCandidate) -> tuple[tuple[int, int], ...]:
        by_task = {entry.task_position: entry.node_position for entry in candidate.result.entries}
        return tuple((int(task), int(by_task[task])) for task in candidate.result.decision_order)

    def _beam_search(
        self,
        scenario: Scenario,
        candidates: list[PortfolioCandidate],
        upward_rank: np.ndarray,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> None:
        """Expand distinct schedules, including non-improving branches, within fixed rounds."""

        unique: dict[tuple[tuple[int, int], ...], PortfolioCandidate] = {}
        for candidate in candidates:
            unique.setdefault(self._schedule_key(candidate), candidate)
        expanded: set[tuple[tuple[int, int], ...]] = set()
        for round_index in range(self.local_search_rounds):
            frontier = sorted(
                (candidate for key, candidate in unique.items() if key not in expanded),
                key=lambda candidate: float(candidate.result.makespan),
            )[:self.local_search_beam_width]
            if not frontier:
                break
            for branch_index, incumbent in enumerate(frontier):
                expanded.add(self._schedule_key(incumbent))
                moves = self._critical_path_node_moves(
                    scenario, incumbent, round_index, execution, communication,
                )
                moves.extend(self._critical_path_task_moves(
                    scenario, incumbent, upward_rank, round_index, execution, communication,
                ))
                for move in moves:
                    self._validate_candidate(move)
                    key = self._schedule_key(move)
                    if key in unique:
                        continue
                    candidate = PortfolioCandidate(f"beam_r{round_index:02d}_b{branch_index:02d}_{move.name}", move.result)
                    unique[key] = candidate
                    candidates.append(candidate)

    @staticmethod
    def _validate_candidate(candidate: PortfolioCandidate) -> None:
        if not np.isfinite(candidate.result.makespan) or candidate.result.makespan <= 0.0:
            raise ValueError(f"candidate {candidate.name!r} produced an invalid makespan")

    def select_task(
        self,
        observation: dict[str, Any],
        ready_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        del observation, deterministic
        if self._cursor >= len(self._decisions):
            raise RuntimeError("portfolio schedule has no remaining task decision")
        task = self._decisions[self._cursor][0]
        if task < 0 or task >= len(ready_mask) or not bool(ready_mask[task]):
            raise RuntimeError(f"preplanned portfolio task {task} is not ready")
        return task

    def select_node(
        self,
        observation: dict[str, Any],
        task_id: int,
        node_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        del observation, deterministic
        if self._cursor >= len(self._decisions):
            raise RuntimeError("portfolio schedule has no remaining node decision")
        expected_task, node = self._decisions[self._cursor]
        if task_id != expected_task:
            raise RuntimeError(f"expected task {expected_task}, received {task_id}")
        if node < 0 or node >= len(node_mask) or not bool(node_mask[node]):
            raise RuntimeError(f"preplanned portfolio node {node} is not feasible")
        self._cursor += 1
        return node

    def _perturbed_heft(
        self,
        scenario: Scenario,
        upward_rank: np.ndarray,
        candidate_index: int,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> SimulationResult:
        simulator = ScheduleSimulator(scenario, execution, communication)
        rng = np.random.default_rng(_stable_seed(self.seed, scenario.scenario_id, candidate_index))
        while not simulator.done:
            ready = sorted(
                simulator.ready_task_positions(),
                key=lambda task: (-float(upward_rank[task]), int(task)),
            )
            task_pool = ready[: min(self.task_top_k, len(ready))]
            # Candidate zero is a task-order-only perturbation; later candidates
            # also explore occasional second-best node placements.
            task_weights = np.exp(-np.arange(len(task_pool), dtype=np.float64))
            task_weights /= task_weights.sum()
            task = int(rng.choice(task_pool, p=task_weights))
            nodes = sorted(
                simulator.feasible_node_positions(task),
                key=lambda node: (simulator.candidate_timing(task, node).finish, int(node)),
            )
            node_pool = nodes[: min(self.node_top_k, len(nodes))]
            choose_alternative = (
                candidate_index > 0
                and len(node_pool) > 1
                and rng.random() < self.alternative_node_probability
            )
            node = int(rng.choice(node_pool[1:])) if choose_alternative else int(node_pool[0])
            simulator.schedule(task, node)
        return simulator.result()

    def _find_discrepancy_points(
        self,
        scenario: Scenario,
        upward_rank: np.ndarray,
        heft_makespan: float,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> tuple[_DiscrepancyPoint, ...]:
        """Select high-criticality, low-regret alternatives on the HEFT trace."""

        if self.task_discrepancy_candidates == 0 and self.node_discrepancy_candidates == 0:
            return ()
        simulator = ScheduleSimulator(scenario, execution, communication)
        rank_scale = max(float(np.max(upward_rank)), 1e-12)
        time_scale = max(float(heft_makespan), 1e-12)
        task_points: list[_DiscrepancyPoint] = []
        node_points: list[_DiscrepancyPoint] = []
        depth = 0
        while not simulator.done:
            ready = sorted(
                simulator.ready_task_positions(),
                key=lambda task: (-float(upward_rank[task]), str(scenario.tasks[task].id)),
            )
            task = int(ready[0])
            criticality = max(float(upward_rank[task]) / rank_scale, 0.0)
            if len(ready) > 1:
                rank_gap = max(float(upward_rank[ready[0]] - upward_rank[ready[1]]), 0.0) / rank_scale
                task_points.append(
                    _DiscrepancyPoint(depth, "task", rank_gap / (0.05 + criticality))
                )
            nodes = sorted(
                simulator.feasible_node_positions(task),
                key=lambda node: (simulator.candidate_timing(task, node).finish, int(node)),
            )
            if len(nodes) > 1:
                best_finish = simulator.candidate_timing(task, nodes[0]).finish
                alternative_finish = simulator.candidate_timing(task, nodes[1]).finish
                finish_gap = max(float(alternative_finish - best_finish), 0.0) / time_scale
                node_points.append(
                    _DiscrepancyPoint(depth, "node", finish_gap / (0.05 + criticality))
                )
            simulator.schedule(task, int(nodes[0]))
            depth += 1
        selected = sorted(task_points, key=lambda point: (point.score, point.depth))[
            : self.task_discrepancy_candidates
        ]
        selected.extend(
            sorted(node_points, key=lambda point: (point.score, point.depth))[
                : self.node_discrepancy_candidates
            ]
        )
        return tuple(selected)

    @staticmethod
    def _single_discrepancy(
        scenario: Scenario,
        upward_rank: np.ndarray,
        point: _DiscrepancyPoint,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> SimulationResult:
        """Replay HEFT with exactly one deterministic task or node deviation."""

        simulator = ScheduleSimulator(scenario, execution, communication)
        depth = 0
        while not simulator.done:
            ready = sorted(
                simulator.ready_task_positions(),
                key=lambda task: (-float(upward_rank[task]), str(scenario.tasks[task].id)),
            )
            if depth == point.depth and point.kind == "task":
                if len(ready) < 2:
                    raise RuntimeError("recorded task discrepancy is no longer available")
                task = int(ready[1])
            else:
                task = int(ready[0])
            nodes = sorted(
                simulator.feasible_node_positions(task),
                key=lambda node: (simulator.candidate_timing(task, node).finish, int(node)),
            )
            if depth == point.depth and point.kind == "node":
                if len(nodes) < 2:
                    raise RuntimeError("recorded node discrepancy is no longer available")
                node = int(nodes[1])
            else:
                node = int(nodes[0])
            simulator.schedule(task, node)
            depth += 1
        return simulator.result()

    def _prefix_template(
        self,
        incumbent: SimulationResult,
        depth: int,
        empty_template: ScheduleSimulator,
    ) -> ScheduleSimulator:
        """Cache requested prefixes for one parent, shared by both move families."""

        if not self.local_search_prefix_cache:
            return empty_template
        if self._prefix_incumbent is not incumbent:
            self._prefix_incumbent = incumbent
            self._prefix_states = {0: empty_template.clone()}
        if depth not in self._prefix_states:
            start = max(position for position in self._prefix_states if position < depth)
            simulator = self._prefix_states[start].clone()
            by_task = {entry.task_position: entry.node_position for entry in incumbent.entries}
            for task in incumbent.decision_order[start:depth]:
                simulator.schedule(task, by_task[task])
            self._prefix_states[depth] = simulator
        return self._prefix_states[depth]

    def _critical_path_node_moves(
        self,
        scenario: Scenario,
        incumbent: PortfolioCandidate,
        round_index: int,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> list[PortfolioCandidate]:
        """Try alternative placements for tasks on the realized critical chain."""

        if self.local_search_critical_tasks == 0:
            return []
        critical_chain = self._realized_critical_chain(scenario, incumbent.result, communication)
        feasibility_simulator = ScheduleSimulator(scenario, execution, communication)
        movable = [
            task
            for task in critical_chain
            if len(feasibility_simulator.feasible_node_positions(task)) > 1
        ]
        if not movable:
            return []
        if len(movable) > self.local_search_critical_tasks:
            indices = np.linspace(
                0,
                len(movable) - 1,
                self.local_search_critical_tasks,
                dtype=int,
            )
            movable = [movable[int(index)] for index in indices]

        moves: list[PortfolioCandidate] = []
        for task in movable:
            depth = incumbent.result.decision_order.index(task)
            prefix = self._prefix_template(incumbent.result, depth, feasibility_simulator)
            for alternative_rank in range(self.local_search_node_alternatives):
                for greedy_repair in (False, True):
                    result = self._replay_with_node_move(
                        scenario,
                        incumbent.result,
                        task,
                        alternative_rank,
                        greedy_repair,
                        execution,
                        communication,
                        prefix,
                    )
                    if result is None:
                        continue
                    repair = "greedy" if greedy_repair else "preserve"
                    moves.append(
                        PortfolioCandidate(
                            f"local_r{round_index:02d}_task{task:03d}_alt{alternative_rank + 1}_{repair}",
                            result,
                        )
                    )
        return moves

    def _critical_path_task_moves(
        self,
        scenario: Scenario,
        incumbent: PortfolioCandidate,
        upward_rank: np.ndarray,
        round_index: int,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> list[PortfolioCandidate]:
        """Try precedence-safe task-order deviations near the realized chain."""

        if self.local_search_critical_tasks == 0 or self.local_search_task_alternatives == 0:
            return []
        critical_chain = list(self._realized_critical_chain(scenario, incumbent.result, communication))
        if len(critical_chain) > self.local_search_critical_tasks:
            indices = np.linspace(
                0,
                len(critical_chain) - 1,
                self.local_search_critical_tasks,
                dtype=int,
            )
            critical_chain = [critical_chain[int(index)] for index in indices]
        depth_by_task = {
            int(task): depth for depth, task in enumerate(incumbent.result.decision_order)
        }
        simulator_template = ScheduleSimulator(scenario, execution, communication)
        moves: list[PortfolioCandidate] = []
        target_depths = sorted(
            {
                depth
                for target_task in critical_chain
                for depth in (
                    depth_by_task[int(target_task)],
                    max(depth_by_task[int(target_task)] - 1, 0),
                )
            }
        )
        for target_depth in target_depths:
            prefix = self._prefix_template(incumbent.result, target_depth, simulator_template)
            for alternative_rank in range(self.local_search_task_alternatives):
                for greedy_repair in (False, True):
                    result = self._replay_with_task_move(
                        scenario,
                        incumbent.result,
                        target_depth,
                        alternative_rank,
                        greedy_repair,
                        upward_rank,
                        execution,
                        communication,
                        prefix,
                    )
                    if result is None:
                        continue
                    repair = "greedy" if greedy_repair else "preserve"
                    moves.append(
                        PortfolioCandidate(
                            f"local_task_r{round_index:02d}_depth{target_depth:03d}_"
                            f"alt{alternative_rank + 1}_{repair}",
                            result,
                        )
                    )
        return moves

    @staticmethod
    def _realized_critical_chain(
        scenario: Scenario,
        result: SimulationResult,
        communication: MatrixCommunicationModel,
    ) -> tuple[int, ...]:
        """Trace one deterministic precedence/resource chain ending at makespan."""

        by_task = {entry.task_position: entry for entry in result.entries}
        timelines: dict[int, list[Any]] = {}
        for entry in result.entries:
            timelines.setdefault(entry.node_position, []).append(entry)
        for timeline in timelines.values():
            timeline.sort(key=lambda entry: (entry.start, entry.finish, entry.task_position))

        current = min(
            result.entries,
            key=lambda entry: (-float(entry.finish), int(entry.task_position)),
        ).task_position
        backwards = [current]
        seen = {current}
        tolerance = 1e-8 * max(1.0, float(result.makespan))
        while True:
            entry = by_task[current]
            causes: list[tuple[float, int]] = []
            for parent in scenario.cache.predecessors[current]:
                parent_entry = by_task[parent]
                arrival = parent_entry.finish + communication.duration(
                    scenario,
                    parent,
                    current,
                    parent_entry.node_position,
                    entry.node_position,
                )
                causes.append((float(arrival), int(parent)))
            previous = [
                other
                for other in timelines[entry.node_position]
                if other.task_position != current and other.finish <= entry.start + tolerance
            ]
            if previous:
                blocker = max(previous, key=lambda other: (other.finish, -other.task_position))
                causes.append((float(blocker.finish), int(blocker.task_position)))
            if not causes:
                break
            cause_time, predecessor = max(causes, key=lambda item: (item[0], -item[1]))
            if cause_time < entry.start - tolerance or predecessor in seen:
                break
            backwards.append(predecessor)
            seen.add(predecessor)
            current = predecessor
        return tuple(reversed(backwards))

    def _critical_block_moves(
        self,
        scenario: Scenario,
        incumbent: PortfolioCandidate,
        round_index: int,
        execution: Any,
        communication: MatrixCommunicationModel,
    ) -> list[PortfolioCandidate]:
        """Rank short critical-chain blocks and try feasible joint co-location."""

        chain = self._realized_critical_chain(scenario, incumbent.result, communication)
        template = ScheduleSimulator(scenario, execution, communication)
        entries = {entry.task_position: entry for entry in incumbent.result.entries}
        blocks: list[tuple[float, tuple[int, ...], list[int]]] = []
        for size in range(2, min(self.local_search_block_max_size, len(chain)) + 1):
            for start in range(len(chain) - size + 1):
                block = chain[start:start + size]
                common = set(template.feasible_node_positions(block[0]))
                for task in block[1:]:
                    common.intersection_update(template.feasible_node_positions(task))
                nodes = [node for node in sorted(common) if any(entries[task].node_position != node for task in block)]
                if not nodes:
                    continue

                def node_cost(node: int) -> float:
                    return sum(
                        float(template.execution_times[task, node]) + sum(
                            communication.duration(scenario, parent, task, entries[parent].node_position, node)
                            for parent in scenario.cache.predecessors[task] if parent not in block
                        )
                        for task in block
                    )

                nodes.sort(key=lambda node: (node_cost(node), node))
                current_cost = sum(
                    entries[task].finish - entries[task].start + sum(
                        communication.duration(scenario, parent, task, entries[parent].node_position, entries[task].node_position)
                        for parent in scenario.cache.predecessors[task]
                    )
                    for task in block
                )
                blocks.append((float(current_cost - node_cost(nodes[0])), block, nodes))
        blocks.sort(key=lambda item: (-item[0], item[1]))
        moves: list[PortfolioCandidate] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for block_index, (_, block, nodes) in enumerate(blocks[:self.local_search_blocks]):
            depth = min(incumbent.result.decision_order.index(task) for task in block)
            prefix = self._prefix_template(incumbent.result, depth, template)
            for node in nodes[:self.local_search_block_node_alternatives]:
                for greedy_repair in (False, True):
                    result = self._replay_with_block_move(incumbent.result, block, node, greedy_repair, prefix)
                    repair = "greedy" if greedy_repair else "preserve"
                    candidate = PortfolioCandidate(
                        f"block_r{round_index:02d}_b{block_index:02d}_size{len(block)}_node{node}_{repair}", result,
                    )
                    key = self._schedule_key(candidate)
                    if key not in seen:
                        seen.add(key)
                        moves.append(candidate)
        return moves

    def _replay_with_block_move(
        self,
        incumbent: SimulationResult,
        block: tuple[int, ...],
        target_node: int,
        greedy_repair: bool,
        prefix: ScheduleSimulator,
    ) -> SimulationResult:
        simulator = prefix.clone()
        placements = {entry.task_position: entry.node_position for entry in incumbent.entries}
        if self.local_search_result_cache and not greedy_repair:
            decisions = tuple(
                (int(task), target_node if task in block else int(placements[task]))
                for task in incumbent.decision_order
            )
            cached = self._cached_replay(decisions, len(simulator.entries))
            if cached is not None:
                return cached
        changed = False
        for task in incumbent.decision_order[len(simulator.entries):]:
            if task in block:
                node = target_node
                changed = True
            elif changed and greedy_repair:
                node = min(
                    simulator.feasible_node_positions(task),
                    key=lambda position: (simulator.candidate_timing(task, position).finish, int(position)),
                )
            else:
                node = int(placements[task])
            simulator.schedule(task, node)
        return self._remember_replay(simulator.result())

    def _remember_replay(self, result: SimulationResult) -> SimulationResult:
        if self.local_search_result_cache:
            key = self._schedule_key(PortfolioCandidate("replay", result))
            self._replay_results.setdefault(key, result)
        return result

    def _cached_replay(
        self,
        decisions: tuple[tuple[int, int], ...],
        prefix_depth: int,
    ) -> SimulationResult | None:
        result = self._replay_results.get(decisions)
        if result is not None:
            self.replay_cache_hits += 1
            self.replay_cache_skipped_tasks += len(decisions) - prefix_depth
        return result

    def _replay_with_node_move(
        self,
        scenario: Scenario,
        incumbent: SimulationResult,
        target_task: int,
        alternative_rank: int,
        greedy_repair: bool,
        execution: Any,
        communication: MatrixCommunicationModel,
        simulator_template: ScheduleSimulator | None = None,
    ) -> SimulationResult | None:
        """Replay a precedence-feasible order with one node change and repair."""

        simulator = (
            simulator_template.clone()
            if simulator_template is not None
            else ScheduleSimulator(scenario, execution, communication)
        )
        incumbent_nodes = {entry.task_position: entry.node_position for entry in incumbent.entries}
        changed = False
        for task in incumbent.decision_order[len(simulator.entries):]:
            feasible = simulator.feasible_node_positions(task)
            if task == target_task:
                alternatives = sorted(
                    (node for node in feasible if node != incumbent_nodes[task]),
                    key=lambda node: (simulator.candidate_timing(task, node).finish, int(node)),
                )
                if alternative_rank >= len(alternatives):
                    return None
                node = int(alternatives[alternative_rank])
                if self.local_search_result_cache and not greedy_repair:
                    decisions = tuple(
                        (int(position), node if position == target_task else int(incumbent_nodes[position]))
                        for position in incumbent.decision_order
                    )
                    cached = self._cached_replay(decisions, len(simulator.entries))
                    if cached is not None:
                        return cached
                changed = True
            elif changed and greedy_repair:
                node = min(
                    feasible,
                    key=lambda item: (simulator.candidate_timing(task, item).finish, int(item)),
                )
            else:
                node = int(incumbent_nodes[task])
            simulator.schedule(task, node)
        return self._remember_replay(simulator.result())

    def _replay_with_task_move(
        self,
        scenario: Scenario,
        incumbent: SimulationResult,
        target_depth: int,
        alternative_rank: int,
        greedy_repair: bool,
        upward_rank: np.ndarray,
        execution: Any,
        communication: MatrixCommunicationModel,
        simulator_template: ScheduleSimulator | None = None,
    ) -> SimulationResult | None:
        """Move one later ready task forward and replay/repair the remaining order."""

        simulator = (
            simulator_template.clone()
            if simulator_template is not None
            else ScheduleSimulator(scenario, execution, communication)
        )
        order = tuple(int(task) for task in incumbent.decision_order)
        original_position = {task: depth for depth, task in enumerate(order)}
        incumbent_nodes = {entry.task_position: entry.node_position for entry in incumbent.entries}
        cursor = len(simulator.entries)
        changed = False
        for depth in range(len(simulator.entries), scenario.num_tasks):
            while cursor < len(order) and order[cursor] in simulator.entries:
                cursor += 1
            if cursor >= len(order):
                return None
            expected = order[cursor]
            if depth == target_depth:
                alternatives = sorted(
                    (task for task in simulator.ready_task_positions() if task != expected),
                    key=lambda task: (
                        -float(upward_rank[task]),
                        original_position[task],
                        int(task),
                    ),
                )
                if alternative_rank >= len(alternatives):
                    return None
                task = int(alternatives[alternative_rank])
                if self.local_search_result_cache and not greedy_repair:
                    moved_order = order[:depth] + (task,) + tuple(
                        position for position in order[depth:] if position != task
                    )
                    decisions = tuple((position, int(incumbent_nodes[position])) for position in moved_order)
                    cached = self._cached_replay(decisions, len(simulator.entries))
                    if cached is not None:
                        return cached
                changed = True
            else:
                task = int(expected)
                cursor += 1
            feasible = simulator.feasible_node_positions(task)
            if changed and greedy_repair:
                node = min(
                    feasible,
                    key=lambda item: (simulator.candidate_timing(task, item).finish, int(item)),
                )
            else:
                node = int(incumbent_nodes[task])
            simulator.schedule(task, node)
        return self._remember_replay(simulator.result())

    @staticmethod
    def _rollout_policy(scenario: Scenario, policy: SchedulerPolicy, normalize_observations: bool = False) -> SimulationResult:
        execution = execution_model_for_scenario(scenario.dataset_source)
        env = CloudEdgeEndDAGEnv(execution, MatrixCommunicationModel(), normalize_observations=normalize_observations)
        high_observation, _ = env.reset(scenario)
        policy.reset(scenario)
        terminated = False
        while not terminated:
            ready = env.get_ready_mask()
            task = policy.select_task(high_observation, ready, deterministic=True)
            env.select_task(task)
            low_observation = env.get_low_observation(task)
            node = policy.select_node(low_observation, task, env.get_node_mask(task), deterministic=True)
            high_observation, _, terminated, truncated, _ = env.step(task, node)
            if truncated:
                raise RuntimeError("offline portfolio candidate cannot truncate")
        return env.simulator.result()
