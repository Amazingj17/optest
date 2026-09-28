"""Normalize saved checkpoint config hashes without changing model state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cpn_hrl_dag.utils.config import config_hash


def normalize(path: Path) -> str:
    """Atomically replace a checkpoint after canonicalizing its config hash."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint has no configuration mapping: {path}")
    canonical = config_hash(config)
    payload["config_hash"] = canonical
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"{checkpoint}: {normalize(checkpoint)}")


if __name__ == "__main__":
    main()
