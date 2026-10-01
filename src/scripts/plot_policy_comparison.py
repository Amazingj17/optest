"""Plot an auditable policy-quality and inference-time comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = (
    ("HEFT", "eval_heft_{split}"),
    ("Greedy EFT", "eval_greedy_eft_{split}"),
    ("Random", "eval_random_{split}"),
    ("Residual HRL", "eval_hrl_{split}"),
    ("HEFT-safe portfolio", "eval_portfolio_{split}"),
    ("HRL-safe", "eval_hrl_safe_{split}"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    root = Path(args.output_dir)
    rows: list[dict[str, float | str]] = []
    num_scenarios = 0
    for label, pattern in POLICIES:
        source = root / pattern.format(split=args.split) / "summary.json"
        if not source.is_file():
            continue
        summary = json.loads(source.read_text(encoding="utf-8"))
        num_scenarios = int(summary["num_scenarios"])
        interval = summary.get("mean_ratio_ci95", [summary["mean_ratio"], summary["mean_ratio"]])
        rows.append(
            {
                "policy": label,
                "mean_ratio": float(summary["mean_ratio"]),
                "ci95_low": float(interval[0]),
                "ci95_high": float(interval[1]),
                "max_ratio": float(summary["max_ratio"]),
                "valid_schedule_rate": float(summary["valid_schedule_rate"]),
                "mean_inference_time_ms": float(summary["mean_inference_time_ms"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no {args.split} policy summaries found below {root}")
    data = pd.DataFrame(rows)
    data.to_csv(root / f"policy_comparison_{args.split}.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required for comparison plots") from error

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    positions = np.arange(len(data))
    errors = np.vstack((data.mean_ratio - data.ci95_low, data.ci95_high - data.mean_ratio))
    colors = ["#6b7280" if value >= 1.0 else "#237a57" for value in data.mean_ratio]
    axes[0].barh(positions, data.mean_ratio, xerr=errors, color=colors, alpha=0.9, capsize=3)
    axes[0].axvline(1.0, color="#1f2937", linestyle="--", linewidth=1)
    axes[0].set_yticks(positions, data.policy)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean makespan / HEFT makespan (95% bootstrap CI)")
    axes[0].set_title("Scheduling quality — lower is better")
    for position, value in zip(positions, data.mean_ratio, strict=True):
        axes[0].text(value + 0.015, position, f"{value:.3f}", va="center", fontsize=9)

    tradeoff = data[data.mean_ratio < 1.2]
    axes[1].scatter(tradeoff.mean_inference_time_ms, tradeoff.mean_ratio, s=60, color="#2a6fbb")
    axes[1].axhline(1.0, color="#1f2937", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Mean policy inference time (ms, log scale)")
    axes[1].set_ylabel("Mean makespan / HEFT makespan")
    axes[1].set_title("Competitive quality–latency trade-off")
    offsets = {
        "HEFT": (5, -15),
        "Greedy EFT": (5, 5),
        "Random": (5, 5),
        "Residual HRL": (5, 8),
        "HEFT-safe portfolio": (-92, -15),
        "HRL-safe": (-18, -18),
    }
    for row in tradeoff.itertuples(index=False):
        axes[1].annotate(row.policy, (row.mean_inference_time_ms, row.mean_ratio), xytext=offsets[row.policy], textcoords="offset points", fontsize=8)
    axes[1].set_xlim(float(tradeoff.mean_inference_time_ms.min()) * 0.8, float(tradeoff.mean_inference_time_ms.max()) * 1.5)
    axes[1].set_ylim(float(tradeoff.mean_ratio.min()) - 0.055, float(tradeoff.mean_ratio.max()) + 0.055)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"CPN-HRL-DAG {args.split}: {num_scenarios} paired scenarios")
    figure.tight_layout()
    target = root / f"policy_comparison_{args.split}.png"
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
