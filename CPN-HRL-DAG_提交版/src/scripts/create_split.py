"""Create a topology-leakage-safe split manifest from audited raw datasets."""
from __future__ import annotations
import argparse
from pathlib import Path
from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.split import SplitManager, scenario_base_key

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--grapheonrl-root',required=True); parser.add_argument('--grapheonrl-system-configs'); parser.add_argument('--archive-task-count',type=int,action='append',default=[50,100,300]); parser.add_argument('--output',required=True); parser.add_argument('--seed',type=int,default=7); parser.add_argument('--limit-per-dataset',type=int)
    args=parser.parse_args(); roots={'grapheonrl':args.grapheonrl_root}
    scenarios=load_scenarios(default_registry(args.grapheonrl_system_configs,args.archive_task_count),roots,limit_per_dataset=args.limit_per_dataset)
    SplitManager.create((scenario_base_key(x) for x in scenarios),seed=args.seed).write(args.output)
    print(f'Wrote topology-safe split manifest for {len(scenarios)} scenarios: {Path(args.output)}')
if __name__=='__main__': main()
