"""Write the two stage-one localisation experiment configs.

Both branches answer "where did the late joint-training regression come from?"
by changing exactly one thing at a time from the *same* source checkpoint, with
the *same* remaining scenario order, the *same* coverage structure and the
*same* environment-decision budget:

* branch A: ``high_only_eft``         - high PPO learns, low level fixed at minimum EFT
* branch B: ``low_only_frozen_high``  - high policy frozen, low PPO learns

Neither branch is exact resume.  Model/optimizer state is restored from the
fork checkpoint; the RNG stream, the scenario permutation cursor and the
coverage counters restart from the declared seed.  Actual optimizer-update
counts are recorded per branch, so equal scenario coverage is never reported as
equal computation.

Usage::

    python scripts/prepare_stage1_forks.py --output outputs/stage1_forks_seed2026 --dry-run
    python scripts/prepare_stage1_forks.py --output outputs/stage1_forks_seed2026
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cpn_hrl_dag.algorithms.phases import parse_phase_schedule, summarize_phase_budget  # noqa: E402
from cpn_hrl_dag.utils.config import config_hash, load_config  # noqa: E402

BRANCHES = ('high_only_eft', 'low_only_frozen_high')


def build_branch_config(base, name, output_dir, episodes, coverage_epochs, diagnostics):
    if name not in BRANCHES:
        raise ValueError(f'unknown fork branch: {name!r}')
    config = copy.deepcopy(base)
    config['experiment'] = dict(config.get('experiment', {}))
    config['experiment']['name'] = f"{base.get('experiment', {}).get('name', 'experiment')}_fork_{name}"
    config['training'] = dict(config['training'])
    config['training']['phases'] = [dict(name=name, episodes=int(episodes))]
    if name == 'low_only_frozen_high':
        config['training']['phases'][0]['frozen_high_mode'] = 'deterministic'
    config['training']['coverage_epochs'] = int(coverage_epochs)
    config['training']['low_pretrain_episodes'] = 0
    config['training']['high_train_episodes'] = 0
    config['training']['joint_train_episodes'] = 0
    config['training']['diagnostics'] = bool(diagnostics)
    # The formal comparison protocol accepts exactly
    # `environment: {normalize_observations: ...}`; the base template still
    # carries the default `ready_semantics` / `insertion_scheduling` keys, which
    # `load_fixed_splits` rejects outright.  Behaviour is identical because those
    # values are the defaults; the keys are simply not part of the frozen schema.
    config['environment'] = {'normalize_observations': bool(config.get('environment', {}).get('normalize_observations', True))}
    # Full 108-scenario validation only.  The formal run shipped with
    # proxy_scenarios=0 and `train_main` hard-refuses anything else, so the base
    # template's legacy proxy value must never leak into a fork experiment.
    config['evaluation'] = dict(config.get('evaluation', {}))
    config['evaluation']['proxy_scenarios'] = 0
    config['evaluation']['deterministic'] = True
    # Explicit, never implicit: optimizer state is restored from the checkpoint
    # but both schedulers are rebuilt for this experiment's own step budget.
    config['fork'] = {'scheduler_restore': 'fresh', 'optimizer_restore': 'restore',
                      'require_checkpoint': True, 'expected_source_step': 864}
    config['output_dir'] = str(output_dir)
    return config


def branch_plan(config):
    schedule = parse_phase_schedule(config['training']['phases'])
    budget = summarize_phase_budget(schedule)
    return dict(phases=[dict(name=item['name'], episodes=item['episodes'], frozen_high_mode=item['frozen_high_mode'],
                             high_action=item['policy'].high, low_action=item['policy'].low,
                             trainable=list(item['policy'].resolved_trainable),
                             frozen=list(item['policy'].frozen)) for item in schedule],
                budget=budget)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', required=True, help='Completed epoch-1 HRL checkpoint at step 864')
    parser.add_argument('--output', required=True, help='New directory for the two fork configs; must not exist')
    parser.add_argument('--output-root', help='Directory the two branch runs write into; defaults to --output. '
                                              'Set this when the project lives at a different path on the training host.')
    parser.add_argument('--config', help='Optional override; default is embedded source checkpoint config')
    parser.add_argument('--episodes', type=int, default=1728, help='Environment episodes per branch')
    parser.add_argument('--coverage-epochs', type=int, default=2)
    parser.add_argument('--diagnostics', action='store_true', help='Enable per-episode training diagnostics')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # Keep an absolute path (including a POSIX one such as /data/...) verbatim:
    # resolving it on the generating host would rewrite it and the target host
    # may not even be the same OS.
    output = Path(args.output)
    if not output.is_absolute():
        output = output.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite an existing directory: {output}')
    if args.episodes < 1 or args.coverage_epochs < 1:
        raise ValueError('episodes and coverage epochs must be positive')
    if args.episodes % 864:
        raise ValueError(f'{args.episodes} episodes do not cover the 864-scenario training set a whole number of times')
    if args.episodes // 864 != args.coverage_epochs:
        raise ValueError(f'coverage_epochs={args.coverage_epochs} contradicts {args.episodes} episodes over 864 scenarios')

    source = Path(args.source_checkpoint).resolve() if args.source_checkpoint else None
    if source is not None and not source.is_file():
        raise FileNotFoundError(f'fork source checkpoint not found: {source}')
    import torch
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if payload.get('config_hash') != config_hash(payload['config']):
        raise ValueError('source checkpoint config hash mismatch')
    if int(payload['global_step']) != 864:
        raise ValueError('stage-one branches require the first-epoch checkpoint at step 864')
    base = load_config(args.config) if args.config else copy.deepcopy(payload['config'])
    base.pop('runtime', None)
    source_config_hash = payload['config_hash']

    run_root = Path(args.output_root).expanduser() if args.output_root else output
    plan = dict(source_checkpoint=str(source) if source else None,
                source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
                source_config_hash=source_config_hash,
                source_config_hash_source='embedded checkpoint config',
                base_config=str(Path(args.config).resolve()) if args.config else 'embedded checkpoint config', base_config_hash=config_hash(base),
                config_dir=str(output), run_root=str(run_root), seed=int(base['experiment']['seed']),
                episodes_per_branch=args.episodes, coverage_epochs=args.coverage_epochs,
                scenarios_per_coverage_epoch=864, validation_scenarios=108, test_evaluated=False,
                fork_semantics='checkpoint_initialization_not_exact_resume',
                scheduler_policy='fresh: start from configured learning_rate; optimizer moments restored',
                matched_across_branches='same remaining scenario order, same coverage epochs, same episode/decision budget',
                branches={})
    configs = {}
    for name in BRANCHES:
        config = build_branch_config(base, name, run_root / name, args.episodes, args.coverage_epochs, args.diagnostics)
        plan['branches'][name] = dict(output_dir=str(run_root / name), **branch_plan(config))
        configs[name] = config
    plan['budget_note'] = ('equal episode and environment-decision budget; optimizer updates differ between branches '
                           'and are reported per branch in budget.json and summary.json')
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'fork_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'configs').mkdir()
    for name, config in configs.items():
        path = output / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        print(f'wrote {path}', flush=True)
    fork_argument = f' --fork-from {source}' if source else ''
    print('run each branch with:', flush=True)
    for name in BRANCHES:
        print(f'  python scripts/train_main_comparison.py --config {output / "configs" / (name + ".yaml")}{fork_argument}', flush=True)


if __name__ == '__main__':
    main()
