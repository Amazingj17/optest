"""Create fixed manifests for Mixed IID, cross-source, scale, and domain OOD runs."""
from __future__ import annotations

import argparse

from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.protocols import scale_generalization_split
from cpn_hrl_dag.datasets.split import SplitManager, SplitManifest, scenario_base_key


def _manifest(grouped: dict[str, list], seed: int, protocol: str) -> SplitManifest:
    return SplitManifest(
        tuple(sorted({scenario_base_key(item) for item in grouped['train']})),
        tuple(sorted({scenario_base_key(item) for item in grouped['validation']})),
        tuple(sorted({scenario_base_key(item) for item in grouped['test']})),
        seed,
        protocol=protocol,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=('mixed_iid', 'scale'), required=True)
    parser.add_argument('--grapheonrl-root', required=True); parser.add_argument('--grapheonrl-system-configs')
    parser.add_argument('--output', required=True); parser.add_argument('--seed', type=int, default=7); parser.add_argument('--limit-per-dataset', type=int)
    parser.add_argument('--max-train-tasks', type=int); parser.add_argument('--min-test-tasks', type=int)
    args = parser.parse_args()
    roots = {'grapheonrl': args.grapheonrl_root}
    scenarios = load_scenarios(default_registry(args.grapheonrl_system_configs), roots, limit_per_dataset=args.limit_per_dataset)
    if args.protocol == 'mixed_iid':
        manifest = SplitManager.create((scenario_base_key(item) for item in scenarios), seed=args.seed)
    elif args.protocol == 'scale':
        if args.max_train_tasks is None or args.min_test_tasks is None: parser.error('scale needs --max-train-tasks and --min-test-tasks')
        manifest = _manifest(scale_generalization_split(scenarios, args.max_train_tasks, args.min_test_tasks), args.seed, 'scale_generalization')
    manifest.write(args.output)
    print(f'Wrote {manifest.protocol} manifest to {args.output}')


if __name__ == '__main__':
    main()
