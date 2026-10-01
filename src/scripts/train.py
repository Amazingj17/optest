"""Staged low-level then joint CPN-HRL-DAG training with validation selection."""
from __future__ import annotations
import argparse, csv, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch, yaml
from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer,high_tensors,low_tensors
from cpn_hrl_dag.algorithms.ppo import PPOAgent
from cpn_hrl_dag.datasets.loading import default_registry,load_scenarios
from cpn_hrl_dag.datasets.split import SplitManager,SplitManifest,scenario_base_key
from cpn_hrl_dag.datasets.sampling import CurriculumStage,DatasetBalancedSampler
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator,HEFTCache,summarize,write_report,write_training_curve
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
from cpn_hrl_dag.utils.config import load_config,config_hash
from cpn_hrl_dag.utils.seed import seed_everything
from cpn_hrl_dag.utils.runtime import runtime_metadata
from cpn_hrl_dag.algorithms.bc import BehaviorCloner
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources

def fixed_proxy_scenarios(scenarios, count):
    """Return a deterministic task-size-balanced validation subset."""
    values=list(scenarios); count=int(count)
    if count<=0 or count>=len(values): return values
    groups={}
    for scenario in values: groups.setdefault(scenario.num_tasks,[]).append(scenario)
    selected=[]; per_group=max(1,count//len(groups))
    for task_count in sorted(groups):
        group=sorted(groups[task_count],key=lambda item:item.scenario_id); indices=np.linspace(0,len(group)-1,min(per_group,len(group)),dtype=int); selected.extend(group[int(index)] for index in indices)
    selected_ids={item.scenario_id for item in selected}
    for scenario in sorted(values,key=lambda item:item.scenario_id):
        if len(selected)>=count: break
        if scenario.scenario_id not in selected_ids: selected.append(scenario); selected_ids.add(scenario.scenario_id)
    return selected[:count]

def make_agents(example,config,device):
    normalize=bool(config.get('environment',{}).get('normalize_observations',False)); high_obs,_=CloudEdgeEndDAGEnv(normalize_observations=normalize).reset(example); task_dim=high_obs['task_features'].shape[1]; resource_dim=high_obs['resource_features'].shape[1]
    probe=CloudEdgeEndDAGEnv(normalize_observations=normalize); probe.reset(example); ready=int(probe.get_ready_mask().nonzero()[0][0]); probe.select_task(ready); low=probe.get_low_observation(ready); node_dim=low['node_features'].shape[1]
    high_config=config['model']['high']; low_config=config['model']['low']; high_model=HighLevelLSTMActorCritic(task_dim,resource_dim,high_config['hidden_dim'],bool(high_config.get('use_lstm',True)),bool(high_config.get('heuristic_residual',False)),float(high_config.get('heuristic_weight',8.0)),float(high_config.get('residual_limit',0.5))).to(device); low_model=LowLevelGATActorCritic(node_dim,task_dim,low_config['hidden_dim'],low_config['heads'],bool(low_config.get('use_gat',True)),bool(low_config.get('heuristic_residual',False)),float(low_config.get('heuristic_weight',8.0)),float(low_config.get('residual_limit',0.5))).to(device); p=config['training']['ppo']
    total=max(1,int(config['training'].get('low_pretrain_episodes',0))+int(config['training'].get('high_train_episodes',0))+int(config['training'].get('joint_train_episodes',0)))
    if config['training'].get('phases'):
        # An explicit phase schedule is the authoritative budget; the legacy
        # per-phase counters may deliberately be left at zero.
        total=max(1,sum(int(item['episodes']) for item in config['training']['phases']))
    high_total=low_total=total
    if config['training'].get('phases'):
        from cpn_hrl_dag.algorithms.phases import resolve_phase_schedule, summarize_phase_budget
        calls=summarize_phase_budget(resolve_phase_schedule(config))['optimizer_updates']
        high_total=max(1,calls['high']); low_total=max(1,calls['low'])
    high_optimizer=torch.optim.Adam(high_model.parameters(),lr=config['model']['high']['learning_rate']); low_optimizer=torch.optim.Adam(low_model.parameters(),lr=config['model']['low']['learning_rate'])
    high_scheduler=torch.optim.lr_scheduler.LinearLR(high_optimizer,start_factor=1.0,end_factor=float(p.get('lr_end_factor',0.1)),total_iters=high_total); low_scheduler=torch.optim.lr_scheduler.LinearLR(low_optimizer,start_factor=1.0,end_factor=float(p.get('lr_end_factor',0.1)),total_iters=low_total)
    high=PPOAgent(high_model,high_optimizer,lambda obs:high_tensors(obs,device),high_model.forward,device,p['gamma_high'],p['gae_lambda_high'],p['clip_coef'],p['entropy_coef_high'],max_grad_norm=p['max_grad_norm'],update_epochs=int(p.get('update_epochs',4)),batch_size=int(p.get('batch_size',64)),lr_scheduler=high_scheduler,target_kl=p.get('target_kl'),value_clip_coef=p.get('value_clip_coef'),lr_scheduler_factory=lambda:torch.optim.lr_scheduler.LinearLR(high_optimizer,start_factor=1.0,end_factor=float(p.get('lr_end_factor',0.1)),total_iters=high_total))
    low_agent=PPOAgent(low_model,low_optimizer,lambda obs:low_tensors(obs,device),low_model.forward,device,p['gamma_low'],p['gae_lambda_low'],p['clip_coef'],p['entropy_coef_low'],max_grad_norm=p['max_grad_norm'],update_epochs=int(p.get('update_epochs',4)),batch_size=int(p.get('batch_size',64)),lr_scheduler=low_scheduler,target_kl=p.get('target_kl'),value_clip_coef=p.get('value_clip_coef'),lr_scheduler_factory=lambda:torch.optim.lr_scheduler.LinearLR(low_optimizer,start_factor=1.0,end_factor=float(p.get('lr_end_factor',0.1)),total_iters=low_total))
    return high,low_agent

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--resume'); args=parser.parse_args(); config=load_config(args.config); seed=int(config['experiment']['seed']); seed_everything(seed,disable_cudnn=bool(config.get('device_options',{}).get('disable_cudnn',False)))
    runtime=runtime_metadata(); config['runtime']=runtime
    output=Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True); (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8'); (output/'reproducibility.json').write_text(json.dumps(runtime,indent=2)+'\n',encoding='utf-8')
    dataset=config['dataset']; scenarios=load_scenarios(default_registry(dataset.get('grapheonrl_system_configs'),dataset.get('archive_task_counts')) ,dataset['roots'],limit_per_dataset=dataset.get('limit_per_dataset'))
    audit={'num_scenarios':len(scenarios),'datasets':sorted({item.dataset_source for item in scenarios}),'scenarios':[{'scenario_id':item.scenario_id,'dataset_source':item.dataset_source,'base_dag_id':item.metadata['original_graph_id'],'num_tasks':item.num_tasks,'num_nodes':item.num_nodes} for item in scenarios]}
    (output/'dataset_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    manifest_path=Path(dataset['split_manifest']); manifest=SplitManifest.read(manifest_path) if manifest_path.is_file() else SplitManager.create((scenario_base_key(x) for x in scenarios),seed=seed); manifest.write(manifest_path); manifest.write(output/'split_manifest.json'); splits=SplitManager.apply(scenarios,manifest)
    splits=materialize_resources(splits,config.get('resources'),manifest,seed)
    device=torch.device(config.get('device','cpu'))
    include_heft=bool(config.get('model',{}).get('high',{}).get('include_heft_features',True)); normalize=bool(config.get('environment',{}).get('normalize_observations',False)); high,low=make_agents(splits['train'][0],config,device); trainer=HierarchicalTrainer(high,low,device,HEFTCache(output/'heft_cache'),config.get('reward',{}),include_heft,normalize); evaluator=Evaluator({'include_heft_features':include_heft,'normalize_observations':normalize}); start=time.perf_counter(); history=[]; best=float('inf'); no_improve=0; resume_step=0
    if args.resume:
        restored=HierarchicalTrainer.load(args.resume,high,low,trainer.device); resume_step=int(restored['global_step']); best=float(restored.get('best_validation_ratio',best))
    bc_config=config.get('bc',{})
    if bool(bc_config.get('enabled',False)):
        bc_scenarios=splits['train'][:int(bc_config.get('max_scenarios',len(splits['train'])))]
        result=BehaviorCloner(high.model,low.model,high.optimizer,low.optimizer,device,normalize).fit(bc_scenarios,int(bc_config.get('epochs',1)))
        (output/'bc.json').write_text(json.dumps(asdict(result),indent=2)+'\n',encoding='utf-8')
    evaluation_config=config.get('evaluation',{})
    proxy_validation=fixed_proxy_scenarios(splits['validation'],int(evaluation_config.get('proxy_scenarios',0)))
    proxy_best=float('inf')
    if bool(evaluation_config.get('evaluate_initial',False)) and resume_step==0:
        initial_records,initial_summary=evaluator.evaluate(CPNHRLDAGPolicy(high.model,low.model,trainer.device),splits['validation']); best=float(initial_summary['mean_ratio']); proxy_records,proxy_summary=(initial_records,initial_summary) if len(proxy_validation)==len(splits['validation']) else evaluator.evaluate(CPNHRLDAGPolicy(high.model,low.model,trainer.device),proxy_validation); proxy_best=float(proxy_summary['mean_ratio']); history.append({'step':0,'phase':'initial','reward':0.0,'low_loss':0.0,'high_loss':0.0,'low_kl':0.0,'high_kl':0.0,'proxy_validation_mean_ratio':proxy_best,'validation_mean_ratio':best}); trainer.save(output/'best.pt',high,low,config,0,best); trainer.save(output/'latest.pt',high,low,config,0,best)
    sampling_weights=dataset.get('sampling',{})
    sampler=DatasetBalancedSampler(splits['train'],sampling_weights,seed)
    stages=[CurriculumStage(int(item['min_tasks']),int(item['max_tasks']),int(item['episodes'])) for item in config.get('curriculum',{}).get('stages',[])]
    phase=['low_pretrain']*int(config['training']['low_pretrain_episodes'])+['high_train']*int(config['training'].get('high_train_episodes',0))+['joint']*int(config['training']['joint_train_episodes'])
    evaluation_interval=max(1,int(config['training'].get('evaluation_interval',1)))
    for step,training_phase in enumerate(phase[resume_step:],resume_step+1):
        stage=None
        if stages:
            cursor=step-1
            for candidate in stages:
                if cursor<candidate.episodes: stage=candidate; break
                cursor-=candidate.episodes
            stage=stage or stages[-1]
        scenario=sampler.sample(stage)
        row=trainer._episode(scenario,training_phase); row['step']=step; row['scenario_id']=scenario.scenario_id; history.append(row)
        if step % evaluation_interval == 0 or step == len(phase):
            policy=CPNHRLDAGPolicy(high.model,low.model,trainer.device); proxy_records,proxy_summary=evaluator.evaluate(policy,proxy_validation); proxy_ratio=float(proxy_summary['mean_ratio']); row['proxy_validation_mean_ratio']=proxy_ratio
            if proxy_ratio<proxy_best:
                proxy_best=proxy_ratio; records,summary=(proxy_records,proxy_summary) if len(proxy_validation)==len(splits['validation']) else evaluator.evaluate(policy,splits['validation']); ratio=float(summary['mean_ratio']); row['validation_mean_ratio']=ratio
                if ratio<best: best=ratio; no_improve=0; trainer.save(output/'best.pt',high,low,config,step,best)
                else: no_improve+=1
            else: no_improve+=1
        trainer.save(output/'latest.pt',high,low,config,step,best)
        if no_improve>=int(config['training']['early_stop_patience']): break
    elapsed=time.perf_counter()-start
    if not (output/'best.pt').is_file():
        records,summary=evaluator.evaluate(CPNHRLDAGPolicy(high.model,low.model,trainer.device),splits['validation']); best=float(summary['mean_ratio']); trainer.save(output/'best.pt',high,low,config,resume_step,best)
    with (output/'train_log.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=sorted({k for x in history for k in x})); writer.writeheader(); writer.writerows(history)
    with (output/'validation_log.csv').open('w',newline='',encoding='utf-8') as handle:
        fields=['step','proxy_validation_mean_ratio','validation_mean_ratio']; writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows({name:row.get(name) for name in fields} for row in history)
    write_training_curve(output,history)
    HierarchicalTrainer.load(output/'best.pt',high,low,trainer.device); records,_=evaluator.evaluate(CPNHRLDAGPolicy(high.model,low.model,trainer.device),splits['validation']); report=summarize(records,model='CPN-HRL-DAG',split='validation',seed=seed,config_hash=config_hash(config),training_time_seconds=elapsed,bootstrap_samples=int(config.get('evaluation',{}).get('bootstrap_samples',0))); write_report(output,records,report); print(f"Validation mean_ratio={report['mean_ratio']:.6f}; valid_schedule_rate={report['valid_schedule_rate']:.3f}; output={output}")
if __name__=='__main__': main()
