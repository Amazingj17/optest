"""Visualize fixed-validation gains of the enhanced HEFT-safe search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _read_rows(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "per_scene.csv")
    required = {"scenario_id", "num_tasks", "ratio", "valid_schedule"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not frame["valid_schedule"].astype(bool).all():
        raise ValueError(f"{label} contains an invalid schedule")
    return frame[["scenario_id", "num_tasks", "ratio"]].rename(columns={"ratio": label})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", default="outputs/zenodo_heft_safe_fast_2026")
    parser.add_argument("--search-root", default="outputs/zenodo_heft_safe_search_2026")
    parser.add_argument("--lns-root", default="outputs/zenodo_heft_safe_search_lns2_2026")
    parser.add_argument("--output")
    args = parser.parse_args()

    legacy = Path(args.legacy_root)
    search = Path(args.search_root)
    lns = Path(args.lns_root)
    output = Path(args.output) if args.output else lns / "search_improvement_validation.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    methods = [
        ("HEFT", {"mean_ratio": 1.0, "mean_ratio_ci95": [1.0, 1.0]}),
        ("Legacy portfolio", _read_summary(lns / "eval_portfolio_validation")),
        ("HRL-safe", _read_summary(legacy / "eval_hrl_safe_validation")),
        ("Advanced search", _read_summary(search / "eval_search_validation")),
        ("Search + 2-round LNS", _read_summary(lns / "eval_search_validation")),
    ]
    comparison = pd.DataFrame(
        {
            "method": [name for name, _ in methods],
            "mean_ratio": [float(summary["mean_ratio"]) for _, summary in methods],
            "ci95_low": [float(summary["mean_ratio_ci95"][0]) for _, summary in methods],
            "ci95_high": [float(summary["mean_ratio_ci95"][1]) for _, summary in methods],
        }
    )
    comparison.to_csv(output.with_name("validation_method_comparison.csv"), index=False)

    portfolio_rows = _read_rows(lns / "eval_portfolio_validation", "legacy_portfolio")
    hrl_rows = _read_rows(legacy / "eval_hrl_safe_validation", "hrl_safe")
    search_rows = _read_rows(search / "eval_search_validation", "advanced_search")
    lns_rows = _read_rows(lns / "eval_search_validation", "search_lns")
    paired = portfolio_rows.merge(hrl_rows, on=["scenario_id", "num_tasks"], validate="one_to_one")
    paired = paired.merge(search_rows, on=["scenario_id", "num_tasks"], validate="one_to_one")
    paired = paired.merge(lns_rows, on=["scenario_id", "num_tasks"], validate="one_to_one")
    paired["gain_vs_hrl_safe"] = paired["hrl_safe"] - paired["search_lns"]
    paired.to_csv(output.with_name("paired_lns_vs_hrl_safe.csv"), index=False)
    rng = np.random.default_rng(2026)

    def paired_interval(values: np.ndarray) -> list[float]:
        samples = np.asarray(
            [rng.choice(values, len(values), replace=True).mean() for _ in range(10_000)]
        )
        return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]

    portfolio_gain = (paired["legacy_portfolio"] - paired["search_lns"]).to_numpy()
    hrl_gain = paired["gain_vs_hrl_safe"].to_numpy()
    statistical_comparison = {
        "num_paired_scenarios": len(paired),
        "mean_gain_vs_legacy_portfolio": float(portfolio_gain.mean()),
        "gain_vs_legacy_portfolio_ci95": paired_interval(portfolio_gain),
        "mean_gain_vs_hrl_safe": float(hrl_gain.mean()),
        "gain_vs_hrl_safe_ci95": paired_interval(hrl_gain),
        "wins_vs_hrl_safe": int((hrl_gain > 1e-9).sum()),
        "ties_vs_hrl_safe": int((np.abs(hrl_gain) <= 1e-9).sum()),
        "losses_vs_hrl_safe": int((hrl_gain < -1e-9).sum()),
    }
    output.with_name("statistical_comparison.json").write_text(
        json.dumps(statistical_comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    colors = ["#777777", "#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    x = np.arange(len(comparison))
    lower = comparison["mean_ratio"] - comparison["ci95_low"]
    upper = comparison["ci95_high"] - comparison["mean_ratio"]
    axes[0, 0].bar(x, comparison["mean_ratio"], color=colors, width=0.72)
    axes[0, 0].errorbar(
        x,
        comparison["mean_ratio"],
        yerr=np.vstack((lower, upper)),
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1,
    )
    axes[0, 0].set_xticks(x, comparison["method"], rotation=18, ha="right")
    axes[0, 0].set_ylabel("Validation mean ratio (lower is better)")
    axes[0, 0].set_ylim(0.88, 1.015)
    axes[0, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    for index, value in enumerate(comparison["mean_ratio"]):
        axes[0, 0].text(index, value + 0.004, f"{value:.4f}", ha="center", fontsize=9)
    axes[0, 0].set_title("A. Overall paired objective")

    size_rows = []
    for column, label in [
        ("legacy_portfolio", "Legacy portfolio"),
        ("hrl_safe", "HRL-safe"),
        ("advanced_search", "Advanced search"),
        ("search_lns", "Search + LNS"),
    ]:
        for task_count, group in paired.groupby("num_tasks", sort=True):
            size_rows.append({"method": label, "tasks": task_count, "mean_ratio": group[column].mean()})
    size_frame = pd.DataFrame(size_rows)
    for color, (label, group) in zip(colors[1:], size_frame.groupby("method", sort=False)):
        axes[0, 1].plot(group["tasks"], group["mean_ratio"], marker="o", linewidth=2, color=color, label=label)
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(sorted(paired["num_tasks"].unique()))
    axes[0, 1].set_xlabel("Number of tasks")
    axes[0, 1].set_ylabel("Mean ratio")
    axes[0, 1].set_title("B. Scale breakdown")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].scatter(paired["hrl_safe"], paired["search_lns"], c=paired["num_tasks"], cmap="viridis", alpha=0.75, s=28)
    low = min(float(paired["hrl_safe"].min()), float(paired["search_lns"].min())) - 0.02
    axes[1, 0].plot([low, 1.01], [low, 1.01], linestyle="--", color="#333333", linewidth=1)
    axes[1, 0].set_xlim(low, 1.01)
    axes[1, 0].set_ylim(low, 1.01)
    axes[1, 0].set_xlabel("HRL-safe ratio")
    axes[1, 0].set_ylabel("Search + LNS ratio")
    axes[1, 0].set_title("C. Per-scenario paired comparison")

    gains = hrl_gain
    axes[1, 1].hist(gains, bins=24, color="#E8684A", alpha=0.85, edgecolor="white")
    axes[1, 1].axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].axvline(float(gains.mean()), color="#5B8FF9", linewidth=2, label=f"Mean gain = {gains.mean():.4f}")
    axes[1, 1].set_xlabel("HRL-safe ratio − Search+LNS ratio")
    axes[1, 1].set_ylabel("Scenario count")
    axes[1, 1].set_title("D. Improvement distribution")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("CPN-HRL-DAG safe-search optimization — fixed validation", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
