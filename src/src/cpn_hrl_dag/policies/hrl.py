"""Inference wrapper for the trained two-level actor-critic models."""
from __future__ import annotations
from typing import Any
import numpy as np, torch
from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.scenario.types import Scenario
from .base import SchedulerPolicy


def _zero_residual_output(model: torch.nn.Module, *, low_level: bool) -> bool:
    """Return whether a residual actor's final layer is exactly zero.

    This check is performed once when a policy wrapper is constructed.  It
    enables an exact and much faster HEFT path for the step-zero checkpoint,
    without changing stochastic PPO collection or trained-model inference.
    """

    if not bool(getattr(model, "heuristic_residual", False)):
        return False
    actor = getattr(model, "actor", None)
    output = actor[-1] if low_level and isinstance(actor, torch.nn.Sequential) else actor
    if not isinstance(output, torch.nn.Linear):
        return False
    return bool(torch.count_nonzero(output.weight).item() == 0 and torch.count_nonzero(output.bias).item() == 0)

def _high_tensors(obs:dict[str,Any],device:torch.device)->tuple[torch.Tensor,...]:
    return (torch.as_tensor(obs['task_features'],device=device).unsqueeze(0),torch.as_tensor(obs['task_mask'],device=device).unsqueeze(0),torch.as_tensor(obs['resource_features'],device=device).unsqueeze(0))

def _low_tensors(obs:dict[str,Any],device:torch.device)->tuple[torch.Tensor,...]:
    return (torch.as_tensor(obs['node_features'],device=device).unsqueeze(0),torch.as_tensor(obs['task_features'],device=device).unsqueeze(0),torch.as_tensor(obs['adjacency'],device=device,dtype=torch.bool).unsqueeze(0),torch.as_tensor(obs['edge_features'],device=device).unsqueeze(0))

def min_eft_node_action(observation:dict[str,Any],node_mask:np.ndarray)->int:
    """Fixed minimum-EFT node rule shared by the diagnostic `fixed_low_eft` mode.

    Reuses the environment's exact float64 ``heuristic_eft`` key so the rule
    matches HEFT processor selection, including exact ties (lowest index wins).
    """
    candidates=np.flatnonzero(node_mask)
    if len(candidates)==0: raise ValueError('node mask has no legal action')
    exact=observation.get('heuristic_eft')
    values=np.asarray(exact if exact is not None else observation['node_features'][:,5],dtype=np.float64)
    return int(min((int(index) for index in candidates),key=lambda index:(float(values[index]),index)))

class CPNHRLDAGPolicy(SchedulerPolicy):
    name='cpn_hrl_dag'
    def __init__(self,high:torch.nn.Module,low:torch.nn.Module,device:str|torch.device='cpu')->None:
        self.high,self.low,self.device=high.eval(),low.eval(),torch.device(device)
        self._exact_high_prior=_zero_residual_output(high,low_level=False)
        self._exact_low_prior=_zero_residual_output(low,low_level=True)
    def reset(self,scenario:Scenario)->None:self._task_ids=tuple(str(task.id) for task in scenario.tasks)
    @torch.no_grad()
    def select_task(self,observation:dict[str,Any],ready_mask:np.ndarray,deterministic:bool=True)->int:
        if deterministic and self._exact_high_prior and "heuristic_priority" in observation:
            values=np.asarray(observation["heuristic_priority"],dtype=np.float64); candidates=np.flatnonzero(ready_mask); return int(min(candidates,key=lambda index:(-float(values[index]),self._task_ids[index])))
        logits,_=self.high(*_high_tensors(observation,self.device)); dist=masked_distribution(logits,torch.as_tensor(ready_mask,device=self.device).unsqueeze(0))
        if deterministic:
            values=dist.logits[0].detach().cpu().numpy(); candidates=np.flatnonzero(ready_mask); return int(min(candidates,key=lambda index:(-float(values[index]),self._task_ids[index])))
        return int(dist.sample().item())
    @torch.no_grad()
    def select_node(self,observation:dict[str,Any],task_id:int,node_mask:np.ndarray,deterministic:bool=True)->int:
        del task_id
        if deterministic and self._exact_low_prior and "heuristic_eft" in observation:
            candidates=np.flatnonzero(node_mask); eft=np.asarray(observation["heuristic_eft"],dtype=np.float64); return int(min(candidates,key=lambda index:(float(eft[index]),int(index))))
        logits,_=self.low(*_low_tensors(observation,self.device)); dist=masked_distribution(logits,torch.as_tensor(node_mask,device=self.device).unsqueeze(0)); return int((torch.argmax(dist.logits,1) if deterministic else dist.sample()).item())


# Execution modes used by the read-only stage-one diagnostic entry point.  They
# are evaluation behaviours, never training behaviours: no mode adds search,
# candidate enumeration, or a HEFT fallback to the "pure" checkpoint policy.
EXECUTION_MODES = ('greedy', 'repeat_sample', 'fixed_low_eft', 'greedy_high_sample_low', 'greedy_high_fixed_eft')


class HRLExecutionModePolicy(SchedulerPolicy):
    """Replay one trained checkpoint under an explicitly named execution mode."""

    def __init__(self,high:torch.nn.Module,low:torch.nn.Module,device:str|torch.device='cpu',mode:str='greedy')->None:
        if mode not in EXECUTION_MODES:
            raise ValueError(f'unknown HRL execution mode: {mode!r}')
        self._policy=CPNHRLDAGPolicy(high,low,device); self.mode=mode; self.device=torch.device(device)
        self.high,self.low=self._policy.high,self._policy.low
        self.name=f'cpn_hrl_dag_{mode}'

    def reset(self,scenario:Scenario)->None:
        self._policy.reset(scenario)

    @torch.no_grad()
    def select_task(self,observation:dict[str,Any],ready_mask:np.ndarray,deterministic:bool=True)->int:
        del deterministic
        return self._policy.select_task(observation,ready_mask,deterministic=self.mode in {'greedy','greedy_high_sample_low','greedy_high_fixed_eft'})

    @torch.no_grad()
    def select_node(self,observation:dict[str,Any],task_id:int,node_mask:np.ndarray,deterministic:bool=True)->int:
        del deterministic
        if self.mode in {'fixed_low_eft','greedy_high_fixed_eft'}:
            return min_eft_node_action(observation,node_mask)
        return self._policy.select_node(observation,task_id,node_mask,deterministic=self.mode=='greedy')
