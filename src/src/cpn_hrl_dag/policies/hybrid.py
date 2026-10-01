"""Independent complete-schedule selection: HEFT, search and learned HRL.

The learned candidate never enters the search frontier. Each branch starts from
a fresh environment; the completed schedules are compared only after validation.
This is a composite policy, not a change to the learned policy or its training.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.simulator import SimulationResult
from .base import SchedulerPolicy
from .heuristics import HEFTPolicy
from .portfolio import HEFTSafePortfolioPolicy


@dataclass(frozen=True)
class CompleteCandidate:
    name: str
    result: SimulationResult
    heft_makespan: float
    inference_time_ms: float
    wall_time_ms: float
    decisions: tuple[tuple[int, int], ...]


class IndependentHybridPolicy(SchedulerPolicy):
    """Choose the smallest validated makespan; exact ties favour HEFT then search.

    Invalid candidates raise an error rather than being silently dropped. The
    dominance guarantee requires every candidate to finish successfully under
    the same deterministic simulator, with no timeouts or approximate scores.
    """

    name = 'hrl_independent_search_heft'

    def __init__(self, learned_policy: SchedulerPolicy, *, search_config: dict,
                 seed: int = 2026, normalize_observations: bool = True,
                 reuse_heft_reference: bool = True):
        forbidden = {'learned_policy', 'seed', 'normalize_observations'} & set(search_config)
        if forbidden:
            raise ValueError(f'search configuration contains reserved options: {sorted(forbidden)}')
        self.learned_policy = learned_policy
        self.search_policy = HEFTSafePortfolioPolicy(
            learned_policy=None, seed=seed, normalize_observations=normalize_observations,
            **search_config)
        self.normalize_observations = normalize_observations
        self.reuse_heft_reference = reuse_heft_reference
        self.candidates: tuple[CompleteCandidate, ...] = ()
        self.selected_candidate = ''
        self.selected_makespan = float('inf')
        self.search_selected_candidate = ''
        self.reset_time_ms = 0.0
        self._decisions: tuple[tuple[int, int], ...] = ()
        self._cursor = 0

    @property
    def planned_decisions(self):
        if not self._decisions:
            raise RuntimeError('hybrid policy has no validated plan; call reset first')
        return self._decisions

    def _generate(self, name, policy, scenario, heft_makespan=None):
        wall_start = perf_counter()
        env = CloudEdgeEndDAGEnv(normalize_observations=self.normalize_observations,
                                heft_makespan=heft_makespan)
        observation, _ = env.reset(scenario)
        # Same standalone timing boundary as Evaluator: environment reset and
        # HEFT reference computation are excluded, policy.reset is included.
        inference_start = perf_counter()
        policy.reset(scenario)
        decisions = []
        done = False
        while not done:
            task = policy.select_task(observation, env.get_ready_mask(), deterministic=True)
            env.select_task(task)
            low = env.get_low_observation(task)
            node = policy.select_node(low, task, env.get_node_mask(task), deterministic=True)
            observation, _, done, truncated, info = env.step(task, node)
            if truncated:
                raise RuntimeError(f'{name}: unexpected truncation')
            decisions.append((int(task), int(node)))
        inference_ms = (perf_counter() - inference_start) * 1000
        result = env.simulator.result()  # validates precedence, feasibility and timelines
        if not info['valid_schedule'] or len(decisions) != scenario.num_tasks:
            raise ValueError(f'{name}: incomplete or invalid schedule')
        if not np.isfinite(result.makespan) or result.makespan <= 0:
            raise ValueError(f'{name}: non-finite or non-positive makespan')
        return CompleteCandidate(name, result, float(env.heft_makespan), inference_ms,
                                 (perf_counter() - wall_start) * 1000, tuple(decisions))

    def reset(self, scenario: Scenario):
        started = perf_counter()
        # Clear any previous plan before starting; a failed reset cannot replay
        # an old scenario's schedule.
        self.candidates = ()
        self._decisions = ()
        self._cursor = 0
        self.selected_candidate = ''
        self.selected_makespan = float('inf')
        self.search_selected_candidate = ''
        if self.search_policy.learned_policy is not None:
            raise ValueError('independent search must not receive a learned policy')
        heft = self._generate('heft', HEFTPolicy(), scenario)
        # Reuse only the current scene's independently checked canonical scalar.
        # Each branch still constructs its own environment and complete schedule.
        if not np.isclose(heft.result.makespan, heft.heft_makespan, rtol=0, atol=1e-9):
            raise ValueError('HEFT candidate differs from the canonical reference')
        reference = heft.heft_makespan if self.reuse_heft_reference else None
        search = self._generate('search_blocks', self.search_policy, scenario, reference)
        learned = self._generate('residual_hrl', self.learned_policy, scenario, reference)
        candidates = (heft, search, learned)
        if any(not np.isclose(c.heft_makespan, heft.heft_makespan, rtol=0, atol=1e-9) for c in candidates):
            raise ValueError('candidate HEFT denominators differ')
        if not np.isclose(heft.result.makespan, heft.heft_makespan, rtol=0, atol=1e-9):
            raise ValueError('HEFT candidate differs from the canonical reference')
        best = min(candidates, key=lambda c: c.result.makespan)
        self.candidates = candidates
        self.selected_candidate = best.name
        self.selected_makespan = best.result.makespan
        self.search_selected_candidate = self.search_policy.selected_candidate
        self._decisions = best.decisions
        self.reset_time_ms = (perf_counter() - started) * 1000

    def select_task(self, observation, ready_mask, deterministic=True):
        del observation, deterministic
        if self._cursor >= len(self._decisions):
            raise RuntimeError('hybrid has no remaining task')
        task, _ = self._decisions[self._cursor]
        if not 0 <= task < len(ready_mask) or not ready_mask[task]:
            raise ValueError('hybrid planned task is not ready')
        return task

    def select_node(self, observation, task_id, node_mask, deterministic=True):
        del observation, deterministic
        if self._cursor >= len(self._decisions):
            raise RuntimeError('hybrid has no remaining node')
        task, node = self._decisions[self._cursor]
        if task_id != task or not 0 <= node < len(node_mask) or not node_mask[node]:
            raise ValueError('hybrid planned task/node is inconsistent or infeasible')
        self._cursor += 1
        return node
