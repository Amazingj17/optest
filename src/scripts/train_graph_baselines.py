"""Train a pure graph PPO or coordinated MAPPO baseline on the fixed training split."""

from __future__ import annotations

import argparse

from cpn_hrl_dag.experiments.graph_baselines import train_baseline
from cpn_hrl_dag.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.device is not None:
        config['device'] = args.device
    print(train_baseline(config, args.output))


if __name__ == '__main__':
    main()
