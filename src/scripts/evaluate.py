"""Evaluate HEFT, Random, or a saved CPN-HRL-DAG checkpoint on a fixed split."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer,high_tensors,low_tensors
from cpn_hrl_dag.algorithms.ppo import PPOAgent
from cpn_hrl_dag.datasets.loading import default_registry,load_scenarios
from cpn_hrl_dag.datasets.split import SplitManifest,SplitManager
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator,summarize,write_report
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.policies.heuristics import HEFTPolicy,RandomPolicy,GreedyEFTPolicy
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.utils.config import load_config,config_hash
from cpn_hrl_dag.utils.seed import seed_everything
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--policy',choices=['heft','random','greedy','hrl','portfolio','search','hrl_safe','hrl_search_safe'],default='heft'); parser.add_argument('--checkpoint'); parser.add_argument('--split',choices=['train','validation','test'],default='validation'); args=parser.parse_args(); config=load_config(args.config); seed_everything(int(config['experiment']['seed']),disable_cudnn=bool(config.get('device_options',{}).get('disable_cudnn',False))); d=config['dataset']; scenarios=load_scenarios(default_registry(d.get('grapheonrl_system_configs'),d.get('archive_task_counts')),d['roots'],limit_per_dataset=d.get('limit_per_dataset')); manifest=SplitManifest.read(d['split_manifest']); output_root=Path(config['output_dir']); output_root.mkdir(parents=True,exist_ok=True); (output_root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8'); manifest.write(output_root/'split_manifest.json'); selected=materialize_resources(SplitManager.apply(scenarios,manifest),config.get('resources'),manifest,int(config['experiment']['seed']))[args.split]
    learned_policy=None
    if args.policy in {'hrl','hrl_safe','hrl_search_safe'}:
        if not args.checkpoint: parser.error(f'--policy {args.policy} requires --checkpoint')
        device=torch.device(config.get('device','cpu')); env=CloudEdgeEndDAGEnv(); high_obs,_=env.reset(selected[0]); task=int(env.get_ready_mask().nonzero()[0][0]); env.select_task(task); low_obs=env.get_low_observation(task); p=config['training']['ppo']
        high_config=config['model']['high']; low_config=config['model']['low']; high_model=HighLevelLSTMActorCritic(high_obs['task_features'].shape[1],high_obs['resource_features'].shape[1],high_config['hidden_dim'],bool(high_config.get('use_lstm',True)),bool(high_config.get('heuristic_residual',False)),float(high_config.get('heuristic_weight',8.0)),float(high_config.get('residual_limit',0.5))).to(device); low_model=LowLevelGATActorCritic(low_obs['node_features'].shape[1],high_obs['task_features'].shape[1],low_config['hidden_dim'],low_config['heads'],bool(low_config.get('use_gat',True)),bool(low_config.get('heuristic_residual',False)),float(low_config.get('heuristic_weight',8.0)),float(low_config.get('residual_limit',0.5))).to(device)
        high=PPOAgent(high_model,torch.optim.Adam(high_model.parameters()),lambda obs:high_tensors(obs,device),high_model.forward,device,p['gamma_high'],p['gae_lambda_high'],p['clip_coef'],p['entropy_coef_high']); low=PPOAgent(low_model,torch.optim.Adam(low_model.parameters()),lambda obs:low_tensors(obs,device),low_model.forward,device,p['gamma_low'],p['gae_lambda_low'],p['clip_coef'],p['entropy_coef_low']); HierarchicalTrainer.load(args.checkpoint,high,low,device); learned_policy=CPNHRLDAGPolicy(high_model,low_model,device)
    portfolio_config=config.get('portfolio',{})
    if args.policy in {'portfolio','search','hrl_safe','hrl_search_safe'}:
        search_config=config.get('search',{}) if args.policy in {'search','hrl_search_safe'} else {}
        policy=HEFTSafePortfolioPolicy(perturbation_candidates=int(search_config.get('perturbation_candidates',portfolio_config.get('perturbation_candidates',16))),task_top_k=int(search_config.get('task_top_k',portfolio_config.get('task_top_k',2))),node_top_k=int(search_config.get('node_top_k',portfolio_config.get('node_top_k',2))),alternative_node_probability=float(search_config.get('alternative_node_probability',portfolio_config.get('alternative_node_probability',0.15))),seed=int(config['experiment']['seed']),learned_policy=learned_policy,normalize_observations=bool(config.get('environment',{}).get('normalize_observations',False)),include_peft=bool(search_config.get('include_peft',False)),peft_lookahead_weights=tuple(search_config.get('peft_lookahead_weights',[1.0])),include_cpop=bool(search_config.get('include_cpop',False)),heft_communication_scales=tuple(search_config.get('heft_communication_scales',[])),task_discrepancy_candidates=int(search_config.get('task_discrepancy_candidates',0)),node_discrepancy_candidates=int(search_config.get('node_discrepancy_candidates',0)),local_search_rounds=int(search_config.get('local_search_rounds',0)),local_search_critical_tasks=int(search_config.get('local_search_critical_tasks',0)),local_search_node_alternatives=int(search_config.get('local_search_node_alternatives',1)),local_search_task_alternatives=int(search_config.get('local_search_task_alternatives',0)),local_search_beam_width=int(search_config.get('local_search_beam_width',1)),local_search_prefix_cache=bool(search_config.get('local_search_prefix_cache',True)),local_search_result_cache=bool(search_config.get('local_search_result_cache',False)),local_search_block_rounds=int(search_config.get('local_search_block_rounds',0)),local_search_blocks=int(search_config.get('local_search_blocks',4)),local_search_block_max_size=int(search_config.get('local_search_block_max_size',3)),local_search_block_node_alternatives=int(search_config.get('local_search_block_node_alternatives',2)))
    elif args.policy == 'hrl': policy=learned_policy
    else: policy={'heft':HEFTPolicy(),'random':RandomPolicy(config['experiment']['seed']),'greedy':GreedyEFTPolicy()}[args.policy]
    evaluator=Evaluator({'include_heft_features':bool(config.get('model',{}).get('high',{}).get('include_heft_features',True)),'normalize_observations':bool(config.get('environment',{}).get('normalize_observations',False))}); records,_=evaluator.evaluate(policy,selected); output=Path(config['output_dir'])/f'eval_{args.policy}_{args.split}'; write_report(output,records,summarize(records,model=policy.name,split=args.split,seed=config['experiment']['seed'],config_hash=config_hash(config),bootstrap_samples=int(config.get('evaluation',{}).get('bootstrap_samples',0)))); print(output/'summary.json')
if __name__=='__main__': main()
