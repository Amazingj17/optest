"""Compare trained graph PPO and tier MAPPO on all 108 validation scenarios only."""

from __future__ import annotations

import argparse
import json

from cpn_hrl_dag.experiments.graph_baselines import compare_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    print(json.dumps(compare_checkpoints(args.checkpoints, args.output, args.device), indent=2))


if __name__ == '__main__':
    main()
