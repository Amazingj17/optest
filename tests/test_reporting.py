from __future__ import annotations

import json

import numpy as np
import pytest

from cpn_hrl_dag.evaluation import EvaluationRecord, summarize, write_report


def _record(identifier: str, ratio: float) -> EvaluationRecord:
    return EvaluationRecord(
        scenario_id=identifier,
        base_dag_id=identifier,
        dataset_source="unit",
        num_tasks=2,
        num_edges=1,
        num_nodes=1,
        num_cloud=0,
        num_edge=1,
        num_end=0,
        domain="unit",
        policy="test",
        makespan=ratio,
        heft_makespan=1.0,
        ratio=ratio,
        valid_schedule=True,
        inference_time_ms=1.0,
    )


def test_heft_comparison_categories_are_disjoint() -> None:
    report = summarize(
        [_record("better", 0.99999), _record("equal", 1.0), _record("worse", 1.00001)],
        model="test",
        split="validation",
        seed=1,
        config_hash="test",
    )
    assert report["better_than_heft_rate"] == 1.0 / 3.0
    assert report["equal_to_heft_rate"] == 1.0 / 3.0


@pytest.mark.parametrize('ratios', [
    [1.0] * 107 + [np.nextafter(1.0, np.inf)],
    [1.0] * 108,
    [1.0],
    [0.8, 0.95, 1.0, 1.25],
])
def test_report_histogram_preserves_nearly_constant_ratios(tmp_path, monkeypatch, ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    original_hist = Axes.hist
    histograms = []

    def capture_hist(axis, values, *args, **kwargs):
        result = original_hist(axis, values, *args, **kwargs)
        histograms.append(result[:2])
        return result

    monkeypatch.setattr(Axes, 'hist', capture_hist)
    records = [_record(str(index), float(ratio)) for index, ratio in enumerate(ratios)]
    summary = summarize(records, model='unit', split='validation', seed=1, config_hash='unit')
    existing_figures = plt.get_fignums()
    write_report(tmp_path, records, summary)
    counts, edges = histograms[0]
    assert counts.sum() == len(ratios)
    assert np.isfinite(edges).all()
    assert (np.diff(edges) > 0).all()
    assert (tmp_path / 'ratio_histogram.png').stat().st_size > 1000
    saved = json.loads((tmp_path / 'summary.json').read_text(encoding='utf-8'))
    assert saved['max_ratio'] == max(ratios)
    assert saved['min_ratio'] == min(ratios)
    assert plt.get_fignums() == existing_figures
