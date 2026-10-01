"""Run reproducible CPN-HRL-DAG architectural ablations from one YAML config."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

# These two variants never implemented the semantics their names claim, so a
# result produced under them would be mislabelled.  They are refused instead of
# silently reported, until the stage-two/stage-three work gives them an explicit
# phase schedule that really removes high-level pretraining or the HEFT residual.
MISLEADING_VARIANTS = {
    "no_staged": "only moved low_pretrain_episodes and left high_train_episodes untouched; "
                 "use an explicit training.phases schedule instead",
    "no_heft_features": "only removed the high-level observation column; it never disabled the HEFT "
                        "residual term in either actor, so it cannot represent 'no HEFT residual'",
}


def variant_config(base: dict, name: str, root: Path) -> dict:
    """Return one explicit, serializable ablation configuration."""
    if name in MISLEADING_VARIANTS:
        raise ValueError(f"ablation {name!r} does not implement its advertised semantics: {MISLEADING_VARIANTS[name]}")
    config = copy.deepcopy(base)
    training = config["training"]
    model = config["model"]
    if name == "no_lstm":
        model["high"]["use_lstm"] = False
    elif name == "no_gat":
        model["low"]["use_gat"] = False
    elif name != "full":
        raise ValueError(f"unknown ablation: {name}")
    config.setdefault("experiment", {})["name"] = f"{base.get('experiment', {}).get('name', 'experiment')}_{name}"
    config["output_dir"] = str(root / name)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--variants", nargs="+", default=["full", "no_lstm", "no_gat"])
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base config must be a YAML mapping")
    refused = sorted(set(args.variants) & set(MISLEADING_VARIANTS))
    if refused:
        raise ValueError(f"refusing misleading ablations {refused}: " + "; ".join(MISLEADING_VARIANTS[name] for name in refused))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.variants:
        config = variant_config(base, name, root)
        config_path = root / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, "scripts/train.py", "--config", str(config_path)], check=True)
        summary = json.loads((Path(config["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
        rows.append({"ablation": name, "output_dir": config["output_dir"], "mean_ratio": summary["mean_ratio"], "valid_schedule_rate": summary["valid_schedule_rate"], "num_scenarios": summary["num_scenarios"]})
    with (root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "ablation_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(root / "ablation_summary.json")


if __name__ == "__main__":
    main()
