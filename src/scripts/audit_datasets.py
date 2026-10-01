"""Validate every selected raw source through its actual adapter schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--grapheonrl-root',required=True); parser.add_argument('--grapheonrl-system-configs'); parser.add_argument('--limit-per-dataset',type=int); parser.add_argument('--output')
    args=parser.parse_args(); roots={'grapheonrl':args.grapheonrl_root}; scenarios=load_scenarios(default_registry(args.grapheonrl_system_configs),roots,limit_per_dataset=args.limit_per_dataset)
    summary={'num_scenarios':len(scenarios),'datasets':{name:sum(x.dataset_source==name for x in scenarios) for name in roots},'scenario_ids':[x.scenario_id for x in scenarios]}
    text=json.dumps(summary,ensure_ascii=False,indent=2)+'\n'; print(text)
    if args.output:
        output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text,encoding='utf-8')
if __name__=='__main__': main()
