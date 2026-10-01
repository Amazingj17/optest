"""Plot validation and online completion-regret diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray(
        [float(row[key]) if row.get(key, "") not in {"", None} else np.nan for row in rows],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Training output directory")
    args = parser.parse_args()
    root = Path(args.output)
    log_path = root / "train_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("training log is empty")

    epochs = _numeric(rows, "epoch")
    validation = _numeric(rows, "validation_mean_ratio")
    actor = _numeric(rows, "actor_loss")
    value = _numeric(rows, "value_loss")
    expected_regret = _numeric(rows, "expected_regret_loss")
    accuracy = _numeric(rows, "action_accuracy")
    improvement = _numeric(rows, "oracle_improvement_rate")
    student_regret = _numeric(rows, "mean_student_regret")
    beta = _numeric(rows, "teacher_beta")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    axes[0, 0].plot(epochs, validation, marker="o", color="#165DFF", linewidth=2)
    best = int(np.nanargmin(validation))
    axes[0, 0].scatter(epochs[best], validation[best], color="#D4380D", zorder=3)
    axes[0, 0].axhline(1.0, color="#6B7280", linestyle="--", linewidth=1, label="HEFT")
    axes[0, 0].set(title="HEFT-safe validation", xlabel="Epoch", ylabel="Mean ratio (lower is better)")
    axes[0, 0].legend()

    for values, label, color in (
        (actor, "Listwise CE", "#165DFF"),
        (value, "Value", "#7A3E9D"),
        (expected_regret, "Expected regret", "#00A870"),
    ):
        if np.isfinite(values).any():
            axes[0, 1].plot(epochs, values, marker="o", label=label, color=color)
    axes[0, 1].set(title="Training objectives", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].legend()

    if np.isfinite(accuracy).any():
        axes[1, 0].plot(epochs, accuracy, marker="o", label="Top-action accuracy", color="#165DFF")
    if np.isfinite(improvement).any():
        axes[1, 0].plot(epochs, improvement, marker="s", label="Oracle differs from HEFT", color="#D4380D")
    axes[1, 0].set(title="Decision diagnostics", xlabel="Epoch", ylabel="Rate", ylim=(-0.02, 1.02))
    axes[1, 0].legend()

    if np.isfinite(student_regret).any():
        axes[1, 1].plot(epochs, student_regret, marker="o", color="#D4380D", label="Student regret")
    beta_axis = axes[1, 1].twinx()
    if np.isfinite(beta).any():
        beta_axis.plot(epochs, beta, marker="s", linestyle="--", color="#6B7280", label="Teacher beta")
    axes[1, 1].set(title="DAgger rollout", xlabel="Epoch", ylabel="Normalized completion regret")
    beta_axis.set_ylabel("Teacher mixture")
    lines = axes[1, 1].get_lines() + beta_axis.get_lines()
    if lines:
        axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="best")

    figure.suptitle("Completion-Regret DAgger Optimization", fontsize=15, fontweight="bold")
    target = root / "dagger_training_curve.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)

    summaries: list[tuple[str, float]] = []
    for label, path in (
        ("Safe search", root / "validation_smoke" / "summary.json"),
        ("Pure graph", root / "validation_pure" / "summary.json"),
    ):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries.append((label, float(payload["mean_ratio"])))
    if summaries:
        comparison, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        labels, values = zip(*summaries)
        bars = axis.bar(labels, values, color=("#165DFF", "#F59E0B")[: len(values)])
        axis.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1, label="HEFT")
        axis.set_ylabel("Validation mean ratio")
        axis.set_title("Best Checkpoint Deployment Comparison")
        axis.bar_label(bars, fmt="%.6f", padding=3)
        axis.legend()
        upper = max(1.02, max(values) * 1.08)
        axis.set_ylim(0.0, upper)
        comparison.savefig(root / "dagger_policy_comparison.png", dpi=180)
        plt.close(comparison)
    print(target)


if __name__ == "__main__":
    main()
