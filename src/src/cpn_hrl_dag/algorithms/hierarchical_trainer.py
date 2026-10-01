"""Staged low-level pretraining then joint hierarchical masked PPO.

One trainer serves every phase.  A phase (:mod:`cpn_hrl_dag.algorithms.phases`)
declares which level samples its own policy, which level follows a fixed rule,
and which level is trainable.  Legacy boolean/`:class:`str` phases keep their
original meaning.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable
import numpy as np, torch
from cpn_hrl_dag.algorithms.phases import DecisionPolicy, canonical_phase_name, decision_policy
from cpn_hrl_dag.algorithms.ppo import PPOAgent,RolloutBuffer
from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation.diagnostics import DecisionDiagnostics
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.execution_model import execution_model_for_scenario
from cpn_hrl_dag.scheduling.communication_model import MatrixCommunicationModel
from cpn_hrl_dag.evaluation.heft_cache import HEFTCache
from cpn_hrl_dag.utils.config import config_hash

def high_tensors(obs,device): return (torch.as_tensor(obs['task_features'],device=device).unsqueeze(0),torch.as_tensor(obs['task_mask'],device=device).unsqueeze(0),torch.as_tensor(obs['resource_features'],device=device).unsqueeze(0))
def low_tensors(obs,device):
    nodes=torch.as_tensor(obs['node_features'],device=device).unsqueeze(0)
    task=torch.as_tensor(obs['task_features'],device=device).unsqueeze(0)
    adjacency=torch.as_tensor(obs['adjacency'],device=device,dtype=torch.bool).unsqueeze(0)
    edges=torch.as_tensor(obs['edge_features'],device=device).unsqueeze(0)
    return nodes,task,adjacency,edges

def fixed_min_eft_node(low_observation:dict[str,Any],node_mask:np.ndarray)->int:
    """Deterministic minimum-EFT node rule shared by every fixed-low phase.

    Delegates to the single canonical rule used by the diagnostic executor so a
    `high_only_eft` training phase and the `fixed_low_eft` diagnostic mode can
    never drift apart.
    """
    from cpn_hrl_dag.policies.hrl import min_eft_node_action
    return min_eft_node_action(low_observation,node_mask)

class HierarchicalTrainer:
    def __init__(self,high:PPOAgent,low:PPOAgent,device:torch.device,heft_cache:HEFTCache|None=None,reward_config:dict|None=None,include_heft_features:bool=True,normalize_observations:bool=False,diagnostics:Callable[[dict[str,Any]],None]|None=None)->None:
        self.high,self.low,self.device,self.heft_cache,self.reward_config,self.include_heft_features,self.normalize_observations=high,low,device,heft_cache,dict(reward_config or {}),bool(include_heft_features),bool(normalize_observations); self.diagnostics=diagnostics
    def _apply_freezing(self,policy:DecisionPolicy)->None:
        """Put every frozen level into deterministic eval mode with no gradients.

        Freezing is not "skip the optimizer": the frozen module must not receive
        gradients, must not change train/eval mode mid-episode, and must never
        advance its learning-rate scheduler.
        """
        for level,agent in (("high",self.high),("low",self.low)):
            if level in policy.frozen:
                if not agent.frozen: agent.freeze()
            elif agent.frozen:
                agent.unfreeze()
    @staticmethod
    def _captured(store:list[dict[str,float]])->Callable[[dict[str,float]],None]:
        """Hook that stores the statistics of the *single* existing forward pass."""
        return store.append
    def _episode(self,scenario:Scenario,phase:str|bool|DecisionPolicy)->dict[str,Any]:
        policy=decision_policy(phase)
        phase_name=policy.name
        self._apply_freezing(policy)
        execution_model=execution_model_for_scenario(scenario.dataset_source)
        communication_model=MatrixCommunicationModel()
        cached_heft=None if self.heft_cache is None else self.heft_cache.get_or_compute(scenario,execution_model,communication_model)
        env=CloudEdgeEndDAGEnv(execution_model,communication_model,heft_makespan=cached_heft,reward_config=self.reward_config,include_heft_features=self.include_heft_features,normalize_observations=self.normalize_observations); high_obs,_=env.reset(scenario); high_buffer,low_buffer=RolloutBuffer(),RolloutBuffer(); ranks=HEFTScheduler(execution_model,communication_model).analyze(scenario).upward_rank; done=False; total=0.0
        diagnostics=DecisionDiagnostics() if self.diagnostics is not None else None
        high_active=policy.update_high; low_active=policy.update_low
        frozen_policy = None
        if policy.high == "ppo_deterministic" and policy.frozen_high_mode == "deterministic":
            from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
            frozen_policy = CPNHRLDAGPolicy(self.high.model, self.low.model, self.device)
            frozen_policy.reset(scenario)
        while not done:
            ready=env.get_ready_mask()
            captured:list[dict[str,float]]=[]
            hook=None if diagnostics is None else self._captured(captured)
            if policy.high=="ppo_sample":
                task,hlog,hval=self.high.act_with_stats(high_obs,ready,trainable=True,hook=hook)
            elif policy.high == "fixed_heft_rank":
                task=int(min(np.flatnonzero(ready),key=lambda x:(-float(ranks[x]),int(x))))
                hlog=hval=0.0
            elif policy.frozen_high_mode=="deterministic":
                # The frozen high level takes its own greedy action; no sample is drawn.
                task=int(frozen_policy.select_task(high_obs,ready,deterministic=True)); hlog,hval=0.0,0.0
            else:
                task,hlog,hval=self.high.act_with_stats(high_obs,ready,trainable=False,hook=hook)
            if diagnostics is not None:
                high_stats=captured.pop() if captured else None
                greedy_high=int(high_stats["argmax_action"]) if high_stats is not None else task
                diagnostics.record_high(ready,greedy_high,task,high_stats,ranks)
            env.select_task(task); low_obs=env.get_low_observation(task); nodes=env.get_node_mask(task)
            diagnostic_eft = low_obs.get('heuristic_eft')
            if diagnostics is not None and diagnostic_eft is not None and not self.normalize_observations:
                diagnostic_eft = diagnostic_eft / max(env.heft_makespan, 1e-9)
            if policy.low=="fixed_min_eft":
                node=fixed_min_eft_node(low_obs,nodes); llog=lval=0.0
                if diagnostics is not None: diagnostics.record_low(nodes,diagnostic_eft,node,None)
            else:
                node,llog,lval=self.low.act_with_stats(low_obs,nodes,trainable=low_active,hook=hook)
                if diagnostics is not None:
                    low_stats=captured.pop() if captured else None
                    diagnostics.record_low(nodes,diagnostic_eft,node,low_stats)
            next_obs,reward,done,_,info=env.step(task,node); total+=reward
            if policy.low=="ppo_sample": low_buffer.add(observation=low_obs,mask=nodes,action=node,log_prob=llog,value=lval,reward=reward,done=done)
            if policy.high=="ppo_sample": high_buffer.add(observation=high_obs,mask=ready,action=task,log_prob=hlog,value=hval,reward=reward,done=done)
            high_obs=next_obs
        decisions=len(high_buffer.items) if policy.high=="ppo_sample" else len(low_buffer.items)
        if diagnostics is not None:
            diagnostics.decisions=decisions
            diagnostics.both_levels_active=decisions if (policy.update_high and policy.update_low) else 0
            diagnostics.high_only_active=decisions if (policy.update_high and not policy.update_low) else 0
            diagnostics.low_only_active=decisions if (policy.update_low and not policy.update_high) else 0
        low_metrics=self.low.update(low_buffer) if low_active else self.frozen_module_metrics()
        high_metrics=self.high.update(high_buffer) if high_active else self.frozen_module_metrics()
        row={"reward":total,"low_loss":low_metrics["loss"],"high_loss":high_metrics["loss"],"low_kl":low_metrics.get("approx_kl"),"high_kl":high_metrics.get("approx_kl"),"phase":phase_name,
             "phase_semantics":phase_name,"valid_schedule":bool(info["valid_schedule"]),"final_makespan":float(info["final_makespan"]),"heft_makespan":float(info["heft_makespan"]),
             "makespan_ratio":float(info["final_makespan"])/max(float(info["heft_makespan"]),1e-9),"decisions":decisions,
             "high_lr":self.high.learning_rate if high_active else None,"low_lr":self.low.learning_rate if low_active else None,
             "high_update_batches":int(high_metrics.get("update_batches",0)),"low_update_batches":int(low_metrics.get("update_batches",0))}
        if diagnostics is not None:
            row.update(self._diagnostic_metrics(diagnostics,low_metrics,high_metrics,policy,row["makespan_ratio"],total))
        if self.diagnostics is not None:
            self.diagnostics(row)
        return row
    @staticmethod
    def frozen_module_metrics()->dict[str,Any]:
        """Explicit "not measured because the module was frozen" metric block."""
        return {"loss":None,"approx_kl":None,"learning_rate":None,"policy_loss":None,"value_loss":None,"entropy":None,
                "clip_fraction":None,"update_batches":0,"grad_norm":None,"train_ratio_mean":None,"train_ratio_max":None,
                "explained_variance":None,"explained_variance_defined":None,"return_variance":None,"frozen":True}
    def _diagnostic_metrics(self,diagnostics:DecisionDiagnostics,low_metrics:dict,high_metrics:dict,policy:DecisionPolicy,makespan_ratio:float,reward:float)->dict[str,Any]:
        aggregated=diagnostics.aggregate()
        metrics={f"diag_{name}":value for name,value in aggregated.items()}
        metrics["diag_makespan_over_heft"]=float(makespan_ratio)
        # With gamma=1, no extra reward terms, and normalized makespan deltas the
        # episode return must equal the negated makespan ratio.  Recorded rather
        # than asserted so a future reward change is visible in the log instead
        # of silently invalidating the stage-one objective.
        metrics["diag_reward_equals_negative_makespan_ratio"]=bool(abs(float(reward)+float(makespan_ratio))<=1e-6)
        for level,prefix,module_metrics in (("high","high",high_metrics),("low","low",low_metrics)):
            frozen=level in policy.frozen
            metrics[f"diag_{prefix}_module_frozen"]=bool(frozen)
            for name,value in module_metrics.items():
                if name in {"learning_rate", "frozen"}: continue
                metrics[f"diag_{prefix}_{name}"]=None if frozen else value
            metrics[f"diag_{prefix}_learning_rate"]=None if frozen else module_metrics.get("learning_rate")
        return metrics
    def train(self,scenarios:Iterable[Scenario],low_pretrain_episodes:int,joint_episodes:int)->list[dict[str,Any]]:
        pool=list(scenarios); assert pool; history=[]
        for episode in range(low_pretrain_episodes+joint_episodes): history.append(self._episode(pool[episode%len(pool)],"low_pretrain" if episode<low_pretrain_episodes else "joint"))
        return history
    @staticmethod
    def save(path:str|Path,high:PPOAgent,low:PPOAgent,config:dict,step:int,best_validation_ratio:float=float('inf'),training_state:dict|None=None)->None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+'.tmp')
        torch.save({'high_model':high.model.state_dict(),'low_model':low.model.state_dict(),'high_optimizer':high.optimizer.state_dict(),'low_optimizer':low.optimizer.state_dict(),'high_lr_scheduler':None if high.lr_scheduler is None else high.lr_scheduler.state_dict(),'low_lr_scheduler':None if low.lr_scheduler is None else low.lr_scheduler.state_dict(),'global_step':step,'config':config,'config_hash':HierarchicalTrainer.config_hash(config),'best_validation_ratio':best_validation_ratio,'training_state':dict(training_state or {})},temporary)
        temporary.replace(path)

    @staticmethod
    def load(path:str|Path,high:PPOAgent,low:PPOAgent,map_location:torch.device|str='cpu')->dict:
        """Restore model and optimizer state, returning persisted experiment facts."""
        payload=torch.load(path,map_location=map_location,weights_only=False)
        required={'high_model','low_model','high_optimizer','low_optimizer','global_step','config'}
        missing=required-set(payload)
        if missing: raise ValueError(f'checkpoint missing keys: {sorted(missing)}')
        high.model.load_state_dict(payload['high_model']); low.model.load_state_dict(payload['low_model'])
        high.optimizer.load_state_dict(payload['high_optimizer']); low.optimizer.load_state_dict(payload['low_optimizer'])
        if high.lr_scheduler is not None and payload.get('high_lr_scheduler') is not None: high.lr_scheduler.load_state_dict(payload['high_lr_scheduler'])
        if low.lr_scheduler is not None and payload.get('low_lr_scheduler') is not None: low.lr_scheduler.load_state_dict(payload['low_lr_scheduler'])
        return payload

    @staticmethod
    def config_hash(config:dict)->str:
        return config_hash(config)
