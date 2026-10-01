"""Training diagnostics for the two-level HRL collector.

Everything here is strictly observational:

* no action is ever sampled twice, so the RNG stream of a training run is
  identical whether diagnostics are enabled or disabled;
* every comparison is made against the *same* environment state that produced
  the executed action, using arrays the environment already built (raw node
  EFT values and the high-level HEFT priority column), never an extra forward
  pass or an extra simulator query;
* modules that a phase freezes report ``None`` for their learning metrics
  instead of a fabricated zero.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DECISION_COUNT_FIELDS = (
    "decisions",
    "both_levels_active",
    "high_only_active",
    "low_only_active",
    "single_legal_action_decisions",
    "multi_legal_action_decisions",
)


def _quantile(values: list[float], probability: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability)) if values else None


@dataclass
class DecisionDiagnostics:
    """Accumulates per-decision statistics for exactly one environment state."""

    high_entropies: list[float] = field(default_factory=list)
    high_expected_log_probabilities: list[float] = field(default_factory=list)
    high_chosen_probabilities: list[float] = field(default_factory=list)
    low_entropies: list[float] = field(default_factory=list)
    low_chosen_probabilities: list[float] = field(default_factory=list)
    low_eft_gaps: list[float] = field(default_factory=list)
    low_deviation_flags: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    decisions: int = 0
    both_levels_active: int = 0
    high_only_active: int = 0
    low_only_active: int = 0
    single_legal_action_decisions: int = 0
    multi_legal_action_decisions: int = 0
    high_deviation_from_greedy: int = 0
    high_rank_preferred_available: int = 0
    high_deviation_from_heft_rank: int = 0
    low_min_eft_tie: int = 0
    low_min_eft_selected: int = 0
    low_suboptimal: int = 0
    low_deviation_from_greedy: int = 0
    low_greedy_comparisons: int = 0
    low_single_legal_action_decisions: int = 0
    low_multi_legal_action_decisions: int = 0
    high_multi_deviation_from_greedy: int = 0
    high_greedy_comparisons: int = 0
    high_multi_greedy_comparisons: int = 0
    low_multi_deviation_from_greedy: int = 0
    low_multi_greedy_comparisons: int = 0
    low_multi_suboptimal: int = 0
    low_multi_eft_comparisons: int = 0
    low_eft_field_missing: int = 0
    high_rank_tie: int = 0
    high_rank_missing: int = 0
    low_stats_missing: int = 0
    high_stats_missing: int = 0
    latest_high_stats: dict[str, float] | None = None
    latest_low_stats: dict[str, float] | None = None

    def record_high(self, ready_mask: np.ndarray, greedy_action: int, chosen_action: int, stats: dict[str, float] | None, rank_priority: np.ndarray | None) -> None:
        self.decisions += 1
        legal = int(np.count_nonzero(ready_mask))
        if legal == 1:
            self.single_legal_action_decisions += 1
        else:
            self.multi_legal_action_decisions += 1
        if stats is not None:
            self.high_greedy_comparisons += 1
            self.high_deviation_from_greedy += int(chosen_action != int(greedy_action))
            if legal > 1:
                self.high_multi_greedy_comparisons += 1
                self.high_multi_deviation_from_greedy += int(chosen_action != int(greedy_action))
        if rank_priority is None:
            self.high_rank_missing += 1
            self.warnings.append("high-level HEFT rank priority was unavailable for a decision")
        else:
            candidates = np.flatnonzero(ready_mask)
            rankings = np.asarray([float(rank_priority[int(index)]) for index in candidates], dtype=np.float64)
            best = float(rankings.max())
            if int(np.count_nonzero(rankings == best)) > 1:
                self.high_rank_tie += 1
            self.high_rank_preferred_available += 1
            if abs(float(rank_priority[int(chosen_action)]) - best) > 1e-12:
                self.high_deviation_from_heft_rank += 1
        if stats is None:
            self.high_stats_missing += 1
        else:
            self.high_entropies.append(float(stats["masked_entropy"]))
            self.high_expected_log_probabilities.append(float(stats["expected_log_prob"]))
            self.high_chosen_probabilities.append(float(stats["chosen_probability"]))

    def record_low(self, node_mask: np.ndarray, heuristic_eft: np.ndarray | None, chosen_action: int, stats: dict[str, float] | None) -> None:
        legal = int(np.count_nonzero(node_mask))
        self.low_single_legal_action_decisions += int(legal == 1)
        self.low_multi_legal_action_decisions += int(legal > 1)
        if stats is not None:
            self.low_greedy_comparisons += 1
            deviated = int(chosen_action != int(stats['argmax_action']))
            self.low_deviation_from_greedy += deviated
            if legal > 1:
                self.low_multi_greedy_comparisons += 1
                self.low_multi_deviation_from_greedy += deviated
        if heuristic_eft is None:
            self.low_eft_field_missing += 1
            self.warnings.append("low-level exact EFT values were missing; the decision is excluded from EFT comparisons")
            if stats is not None:
                self.low_entropies.append(float(stats["masked_entropy"]))
                self.low_chosen_probabilities.append(float(stats["chosen_probability"]))
            self.low_deviation_flags.append(-1)
            return
        min_eft, optimal_count, greedy_action, is_optimal, deviation = min_eft_choice(heuristic_eft, node_mask, chosen_action)
        del min_eft
        chosen = int(chosen_action)
        if optimal_count > 1:
            self.low_min_eft_tie += 1
        if is_optimal:
            self.low_min_eft_selected += 1
        else:
            self.low_suboptimal += 1
        self.low_eft_gaps.append(float(deviation))
        if legal > 1:
            self.low_multi_eft_comparisons += 1
            self.low_multi_suboptimal += int(not is_optimal)
        # "Deviation from the greedy node" counts an exact tie broken the other
        # way as well; "suboptimal" counts only strictly worse EFT choices.
        if chosen == greedy_action:
            self.low_deviation_flags.append(0)
        else:
            self.low_deviation_flags.append(1)
        if stats is None:
            self.low_stats_missing += 1
        else:
            self.low_entropies.append(float(stats["masked_entropy"]))
            self.low_chosen_probabilities.append(float(stats["chosen_probability"]))

    def aggregate(self) -> dict[str, float | int | None]:
        counts = dict(decisions=self.decisions, both_levels_active=self.both_levels_active,
                      high_only_active=self.high_only_active, low_only_active=self.low_only_active,
                      single_legal_action_decisions=self.single_legal_action_decisions,
                      multi_legal_action_decisions=self.multi_legal_action_decisions)
        active = sum(1 for flag in self.low_deviation_flags if flag >= 0)
        result: dict[str, float | int | None] = dict(counts)
        result.update(
            high_entropy_active=_mean(self.high_entropies),
            high_expected_log_probability_active=_mean(self.high_expected_log_probabilities),
            high_chosen_probability_active=_mean(self.high_chosen_probabilities),
            high_deviation_rate_from_greedy=_rate(self.high_deviation_from_greedy, self.high_greedy_comparisons),
            high_deviation_rate_from_greedy_multi=_rate(self.high_multi_deviation_from_greedy, self.high_multi_greedy_comparisons),
            high_deviation_rate_from_heft_rank=_rate(self.high_deviation_from_heft_rank, self.high_rank_preferred_available),
            low_entropy_active=_mean(self.low_entropies),
            low_choice_probability_mean_active=_mean(self.low_chosen_probabilities),
            low_deviation_rate_from_min_eft=_rate(sum(1 for flag in self.low_deviation_flags if flag == 1), active),
            low_deviation_rate_from_greedy=_rate(self.low_deviation_from_greedy, self.low_greedy_comparisons),
            low_deviation_rate_from_greedy_multi=_rate(self.low_multi_deviation_from_greedy, self.low_multi_greedy_comparisons),
            low_single_legal_action_decisions=self.low_single_legal_action_decisions,
            low_multi_legal_action_decisions=self.low_multi_legal_action_decisions,
            low_suboptimal_rate_multi=_rate(self.low_multi_suboptimal, self.low_multi_eft_comparisons),
            low_suboptimal_rate_active=_rate(self.low_suboptimal, active),
            low_selected_eft_gap_mean_active=_mean(self.low_eft_gaps),
            low_selected_eft_gap_p95_active=_quantile(self.low_eft_gaps, .95) if self.low_eft_gaps else None,
            low_selected_eft_gap_max_active=max(self.low_eft_gaps) if self.low_eft_gaps else None,
            low_min_eft_tie_decisions=self.low_min_eft_tie,
            low_min_eft_selected_decisions=self.low_min_eft_selected,
            low_suboptimal_decisions=self.low_suboptimal,
            low_eft_field_missing_decisions=self.low_eft_field_missing,
            high_rank_tie_decisions=self.high_rank_tie,
            high_rank_missing_decisions=self.high_rank_missing,
            high_action_stats_missing_decisions=self.high_stats_missing,
            low_action_stats_missing_decisions=self.low_stats_missing,
            diagnostic_warning_count=len(self.warnings),
        )
        return result


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def min_eft_choice(heuristic_eft: np.ndarray, node_mask: np.ndarray, chosen_action: int) -> tuple[float, int, int, int, float]:
    """Resolve the minimum-EFT choice for one node-selection state.

    Ties are explicit: ``optimal_count`` reports how many legal nodes share the
    exact minimum EFT (float64, so HEFT's own ordering is preserved), and
    ``greedy_action`` is the deterministic lowest-index node among them.  A
    chosen action inside the tied minimum counts as optimal even when it is not
    the lowest index, which is what separates "tied optimum" from "genuinely
    suboptimal" in the aggregated diagnostics.

    Returns ``(minimum_eft, optimal_count, greedy_action, is_optimal, deviation)``.
    """
    candidates = np.asarray(np.flatnonzero(node_mask), dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("diagnostics received a node mask without a legal action")
    values = np.asarray(heuristic_eft, dtype=np.float64)
    minimum = float(np.min(values[candidates]))
    optimal = candidates[values[candidates] <= minimum + 1e-12]
    chosen = int(chosen_action)
    if chosen not in set(int(index) for index in candidates):
        raise ValueError("chosen node action is outside the legal mask")
    greedy_action = int(min((int(index) for index in optimal), key=lambda index: (float(values[index]), index)))
    is_optimal = int(bool(np.any(optimal == chosen)))
    deviation = max(float(values[chosen]) - minimum, 0.0)
    return minimum, int(optimal.size), greedy_action, is_optimal, deviation


def eft_choice_statistics(heuristic_eft: np.ndarray | None, node_mask: np.ndarray, chosen_action: int) -> tuple[int, int, float | None]:
    """Backwards-compatible view of :func:`min_eft_choice` for ``None`` EFT input."""
    if heuristic_eft is None:
        return 0, 0, None
    _, optimal_count, _, is_optimal, deviation = min_eft_choice(heuristic_eft, node_mask, chosen_action)
    return optimal_count, is_optimal, deviation


class DiagnosticsRecorder:
    """Incremental per-episode diagnostics sink (JSONL + flushed CSV summary)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.directory / 'training_diagnostics.jsonl'
        self.csv_path = self.directory / 'training_diagnostics_summary.csv'
        self._jsonl = None
        self._csv = None
        self._writer = None
        self._rows: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []

    def _handle(self):
        if self._jsonl is None:
            self._jsonl = self.jsonl_path.open('x', encoding='utf-8')
        return self._jsonl

    def record(self, row: dict[str, Any]) -> None:
        handle = self._handle()
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        self._rows.append(row)
        if self._csv is None:
            self._csv = self.csv_path.open('x', newline='', encoding='utf-8')
            self._writer = csv.DictWriter(self._csv, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv.flush()

    def capture(self, row: dict[str, Any]) -> None:
        """Trainer callback: stash one episode row until its metadata is known.

        The trainer hands the row over before the surrounding loop adds
        ``episode``/``epoch``/``scenario_id``, so the row is buffered and written
        by :meth:`flush_captured` once those identifying columns exist.  Keeping
        the trainer unaware of file layout is deliberate.
        """
        self._pending.append(row)

    def flush_captured(self, overrides: dict[str, Any]) -> None:
        """Write every stashed row merged with the loop's identifying columns.

        Row values win over the loop metadata, so the trainer's own per-episode
        measurements are never overwritten by a coarser loop counter.
        """
        while self._pending:
            row = self._pending.pop(0)
            merged = dict(overrides)
            merged.update(row)
            self.record(merged)

    def _write_summary(self) -> None:
        if not self._rows:
            return
        preferred = ['step', 'epoch', 'phase', 'scenario_id', 'num_tasks', 'coverage', 'decisions']
        fields = list(preferred)
        for row in self._rows:
            for name in row:
                if name not in fields:
                    fields.append(name)
        temporary = self.csv_path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._rows)
            handle.flush()
        temporary.replace(self.csv_path)

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        if self._csv is not None:
            self._csv.close()
            self._csv = None
        if self._rows:
            phases = {}
            for phase in sorted({row['phase'] for row in self._rows if 'phase' in row}):
                rows = [row for row in self._rows if row.get('phase') == phase]
                means = {}
                for key in rows[0]:
                    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
                              and not isinstance(row.get(key), bool)]
                    if values:
                        means[key] = {'mean': float(np.mean(values)), 'measured_episodes': len(values)}
                phases[phase] = {'episodes': len(rows), 'decisions': sum(row.get('decisions', 0) for row in rows),
                                 'episode_means': means}
            from cpn_hrl_dag.utils.progress import atomic_json
            atomic_json(self.directory / 'diagnostics_by_phase.json', {
                'aggregation': 'unweighted episode means; mean of episode p95 is NOT pooled p95', 'phases': phases})

    def __enter__(self) -> "DiagnosticsRecorder":
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        self.close()
