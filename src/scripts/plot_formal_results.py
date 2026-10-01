"""Render a compact dashboard from an actual CPN-HRL-DAG evaluation export.

The script deliberately consumes ``per_scene.csv`` written by the shared
Evaluator.  It never re-computes schedules and therefore cannot accidentally
mix a different simulator or resource realization into the plotted results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def size_bucket(tasks: int) -> str:
    return "small" if tasks <= 50 else "medium" if tasks <= 100 else "large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    args = parser.parse_args()
    target = Path(args.evaluation_dir)
    source = target / "per_scene.csv"
    if not source.is_file():
        raise FileNotFoundError(f"missing evaluation export: {source}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required for formal result plots") from error

    data = pd.read_csv(source)
    required = {"dataset_source", "num_tasks", "ratio", "heft_makespan", "makespan"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"per_scene.csv misses required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("cannot plot an empty evaluation")
    data["size"] = data["num_tasks"].map(size_bucket)

    summary_path = target / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    model = str(summary.get("model", data["policy"].iloc[0] if "policy" in data else "scheduler"))
    split = str(summary.get("split", "evaluation"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(f"{model}: {split} evaluation", fontsize=14)

    grouped = data.groupby("dataset_source")["ratio"].agg(["mean", "std"]).sort_index()
    axes[0, 0].bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=4, color="#2a6fbb")
    axes[0, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="HEFT")
    axes[0, 0].set(title="Mean ratio by dataset", ylabel="Policy makespan / HEFT makespan")
    axes[0, 0].legend()

    ordered_sizes = [value for value in ("small", "medium", "large") if value in set(data["size"])]
    grouped_size = data.groupby("size")["ratio"].agg(["mean", "std"]).reindex(ordered_sizes)
    axes[0, 1].bar(grouped_size.index, grouped_size["mean"], yerr=grouped_size["std"], capsize=4, color="#36a269")
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="HEFT")
    axes[0, 1].set(title="Mean ratio by DAG size", ylabel="Policy makespan / HEFT makespan")
    axes[0, 1].legend()

    axes[1, 0].hist(data["ratio"], bins=min(30, max(5, len(data) // 4)), color="#a15db0", edgecolor="white")
    axes[1, 0].axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Per-scenario ratio distribution", xlabel="Policy makespan / HEFT makespan", ylabel="scenario count")

    axes[1, 1].scatter(data["heft_makespan"], data["makespan"], s=18, alpha=0.65, color="#d96b27")
    lower = float(min(data["heft_makespan"].min(), data["makespan"].min()))
    upper = float(max(data["heft_makespan"].max(), data["makespan"].max()))
    axes[1, 1].plot([lower, upper], [lower, upper], "--", color="#333333", linewidth=1)
    axes[1, 1].set(title="Makespan pairing", xlabel="HEFT makespan", ylabel=f"{model} makespan")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(target / "formal_test_dashboard.png", dpi=180)
    plt.close(figure)
    print(target / "formal_test_dashboard.png")


if __name__ == "__main__":
    main()
