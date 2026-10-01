"""Masked categorical PPO with GAE and self-contained rollout records."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch
from torch import nn

def masked_distribution(logits:torch.Tensor,mask:torch.Tensor)->torch.distributions.Categorical:
    if not bool(mask.any(-1).all()): raise ValueError("every masked action distribution needs a legal action")
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-1e9))

def explained_variance(values:np.ndarray,returns:np.ndarray)->float:
    """Fraction of return variance explained by the critic.

    Zero-variance returns carry no information for the critic, so the ratio is
    undefined there.  The callers must report that case explicitly instead of
    publishing a fabricated number; this function returns ``0.0`` only for a
    genuinely degenerate target and the *actual* ratio otherwise.
    """
    values=np.asarray(values,dtype=np.float64); returns=np.asarray(returns,dtype=np.float64)
    if values.size==0: return 0.0
    variance=float(np.var(returns))
    if not np.isfinite(variance) or variance<=1e-12: return 0.0
    residual=float(np.var(returns-values))
    if not np.isfinite(residual): return 0.0
    return float(1.0-residual/variance)

@dataclass
class Transition:
    observation:dict[str,np.ndarray]; mask:np.ndarray; action:int; log_prob:float; value:float; reward:float; done:bool

class RolloutBuffer:
    def __init__(self)->None:self.items:list[Transition]=[]
    def add(self,**kwargs:Any)->None:self.items.append(Transition(**kwargs))
    def clear(self)->None:self.items.clear()
    def gae(self,gamma:float,lam:float,last_value:float=0.0)->tuple[np.ndarray,np.ndarray]:
        advantages=np.zeros(len(self.items),np.float32); next_value=last_value; gae=0.0
        for i in range(len(self.items)-1,-1,-1):
            item=self.items[i]; delta=item.reward+gamma*next_value*(1.0-float(item.done))-item.value; gae=delta+gamma*lam*(1.0-float(item.done))*gae; advantages[i]=gae; next_value=item.value
        returns=advantages+np.asarray([item.value for item in self.items],np.float32); return advantages,returns

class PPOAgent:
    def __init__(self,model:nn.Module,optimizer:torch.optim.Optimizer,observation_tensor:Callable[[dict[str,np.ndarray]],tuple[torch.Tensor,...]],forward:Callable[...,tuple[torch.Tensor,torch.Tensor]],device:torch.device,gamma:float,gae_lambda:float,clip_coef:float,entropy_coef:float,value_coef:float=0.5,max_grad_norm:float=1.0,update_epochs:int=4,batch_size:int=64,lr_scheduler:torch.optim.lr_scheduler.LRScheduler|None=None,target_kl:float|None=None,value_clip_coef:float|None=None,lr_scheduler_factory:Callable[[],torch.optim.lr_scheduler.LRScheduler|None]|None=None)->None:
        self.model,self.optimizer,self.observation_tensor,self.forward,self.device=model,optimizer,observation_tensor,forward,device; self.gamma,self.gae_lambda,self.clip_coef,self.entropy_coef,self.value_coef,self.max_grad_norm=gamma,gae_lambda,clip_coef,entropy_coef,value_coef,max_grad_norm; self.update_epochs,self.batch_size,self.lr_scheduler=update_epochs,batch_size,lr_scheduler; self.target_kl=None if target_kl is None else float(target_kl); self.value_clip_coef=None if value_clip_coef is None else float(value_clip_coef)
        self.lr_scheduler_factory=lr_scheduler_factory
        if self.target_kl is not None and self.target_kl <= 0.0: raise ValueError("target_kl must be positive")
        if self.value_clip_coef is not None and self.value_clip_coef <= 0.0: raise ValueError("value_clip_coef must be positive")
        self._frozen=bool(getattr(optimizer,'frozen',False))
    def rebuild_lr_scheduler(self)->None:
        """Replace the scheduler with a fresh one built for this experiment's budget.

        Used when a new experiment forks from a checkpoint whose scheduler was
        built for a different total step count.  Raises when no factory was
        provided, so an implicit scheduler inheritance can never happen quietly.
        """
        if self.lr_scheduler_factory is None:
            raise RuntimeError("no learning-rate scheduler factory was configured; refusing to guess a schedule")
        self.lr_scheduler=self.lr_scheduler_factory()
    @property
    def frozen(self)->bool:
        """Whether this agent is deliberately excluded from every update."""
        return self._frozen
    @property
    def learning_rate(self)->float:
        return float(self.optimizer.param_groups[0]['lr'])
    def freeze(self)->None:
        """Freeze parameters, keep evaluation mode, and block update/scheduler effects.

        Freezing is more than skipping ``optimizer.step``: the module switches to
        deterministic evaluation mode, gradients are disabled, and the learning
        rate scheduler must no longer advance because no parameter changed.
        """
        self._frozen=True
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.model.eval()
    def unfreeze(self)->None:
        self._frozen=False
        for parameter in self.model.parameters(): parameter.requires_grad_(True)
    @torch.no_grad()
    def act_with_stats(self,observation:dict[str,np.ndarray],mask:np.ndarray,deterministic:bool=False,trainable:bool=True,hook:Callable[[dict[str,Any]],None]|None=None)->tuple[int,float,float]:
        """Sample or select one masked action and expose the exact distribution used.

        The returned statistics come from the *same* distribution object that
        generated the action, so diagnostics never re-run a forward pass, never
        draw an extra sample, and never consume randomness.  ``hook`` receives
        the statistics and is the only extension point used by training
        diagnostics.
        """
        if trainable and not self._frozen: self.model.train()
        else: self.model.eval()
        tensors=self.observation_tensor(observation); logits,value=self.forward(*tensors); dist=masked_distribution(logits,torch.as_tensor(mask,device=self.device).unsqueeze(0))
        action=torch.argmax(dist.logits,1) if deterministic else dist.sample()
        log_prob=dist.log_prob(action)
        if hook is not None: hook(self._action_stats(dist,action,logits))
        return int(action.item()),float(log_prob.item()),float(value.item())
    @staticmethod
    def _action_stats(dist:torch.distributions.Categorical,action:torch.Tensor,logits:torch.Tensor)->dict[str,float]:
        """Summarize the exact distribution that produced ``action``.

        Nothing here resamples: ``argmax_action`` is the greedy action *of the
        same forward pass*, which is what the deviation diagnostics compare
        against, and the entropy uses the already-masked logits (with an
        explicit ``0 * log 0 = 0`` convention) so masked actions cannot leak a
        ``NaN`` into a log line.
        """
        probabilities=dist.probs.detach(); masked_logits=dist.logits.detach(); legal=(masked_logits>-1e8)
        safe_log=masked_logits.clamp_min(-1e30)
        masked_entropy=float((-(probabilities*(safe_log.masked_fill(~legal,0.0))).sum(-1)).item())
        return dict(entropy=float(dist.entropy().item()),masked_entropy=masked_entropy,
                    chosen_log_prob=float(masked_logits[0,int(action.item())].item()),chosen_probability=float(probabilities[0,int(action.item())].item()),
                    top_log_prob=float(masked_logits.max(-1).values[0].item()),expected_log_prob=float((probabilities*safe_log.masked_fill(~legal,0.0)).sum(-1).item()),
                    argmax_action=float(masked_logits.argmax(-1)[0].item()),
                    raw_logit_spread=float((logits.detach().max(-1).values-logits.detach().min(-1).values)[0].item()))
    @torch.no_grad()
    def act(self,observation:dict[str,np.ndarray],mask:np.ndarray,deterministic:bool=False)->tuple[int,float,float]:
        return self.act_with_stats(observation,mask,deterministic)
    @torch.no_grad()
    def greedy_action(self,observation:dict[str,np.ndarray],mask:np.ndarray)->int:
        """Return the arg-max action without sampling and without affecting the RNG."""
        return int(self.act_with_stats(observation,mask,True)[0])
    def update(self,buffer:RolloutBuffer,epochs:int|None=None)->dict[str,float]:
        if self._frozen: raise RuntimeError("refusing to update a frozen PPO agent")
        if not buffer.items:return {"loss":0.0}
        # Evaluation policies intentionally switch the shared modules to eval
        # mode.  PPO must restore training mode before its *new* forward pass;
        # in particular CuDNN LSTM backward requires a training-mode forward.
        self.model.train()
        epochs=self.update_epochs if epochs is None else epochs
        advantages,returns=buffer.gae(self.gamma,self.gae_lambda)
        # Captured before the buffer is cleared so explained variance describes
        # exactly the rollouts that were updated.
        rollout_values=np.asarray([item.value for item in buffer.items],np.float32)
        rollout_returns=np.asarray(returns,np.float32)
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8); losses=[]; measured_kl=[]; stop_early=False
        policy_losses=[]; value_losses=[]; entropies=[]; clip_fractions=[]; ratios_all=[]; grad_norms=[]
        for _ in range(epochs):
            order=np.random.permutation(len(buffer.items))
            for start in range(0,len(order),self.batch_size):
                batch=order[start:start+self.batch_size]; self.optimizer.zero_grad(); total_loss=torch.zeros((),device=self.device); batch_kl=torch.zeros((),device=self.device)
                batch_policy=torch.zeros((),device=self.device); batch_value=torch.zeros((),device=self.device); batch_entropy=torch.zeros((),device=self.device); batch_clip=torch.zeros((),device=self.device); batch_ratio=torch.zeros((),device=self.device)
                # Variable-size DAGs are grouped by exact tensor shapes.  Each
                # group is one batched LSTM/GAT forward rather than one forward
                # per transition, while masks remain exactly those sampled.
                groups:dict[tuple[tuple[int,...],...],list[tuple[int,tuple[torch.Tensor,...]]]]={}
                for index in batch:
                    tensors=self.observation_tensor(buffer.items[int(index)].observation)
                    signature=tuple(tuple(tensor.shape[1:]) for tensor in tensors)
                    groups.setdefault(signature,[]).append((int(index),tensors))
                for group in groups.values():
                    indices=[item[0] for item in group]; arguments=tuple(torch.cat([item[1][position] for item in group],dim=0) for position in range(len(group[0][1])))
                    logits,value=self.forward(*arguments); masks=torch.as_tensor(np.stack([buffer.items[index].mask for index in indices]),device=self.device,dtype=torch.bool); dist=masked_distribution(logits,masks); actions=torch.as_tensor([buffer.items[index].action for index in indices],device=self.device,dtype=torch.long); old_log_prob=torch.as_tensor([buffer.items[index].log_prob for index in indices],device=self.device,dtype=torch.float32); advantage=torch.as_tensor(advantages[indices],device=self.device,dtype=torch.float32); target_return=torch.as_tensor(returns[indices],device=self.device,dtype=torch.float32); old_value=torch.as_tensor([buffer.items[index].value for index in indices],device=self.device,dtype=torch.float32)
                    new_log_prob=dist.log_prob(actions); log_ratio=new_log_prob-old_log_prob; ratio=torch.exp(log_ratio); policy=-torch.min(ratio*advantage,torch.clamp(ratio,1-self.clip_coef,1+self.clip_coef)*advantage)
                    value_error=(value-target_return).pow(2)
                    if self.value_clip_coef is not None:
                        clipped_value=old_value+(value-old_value).clamp(-self.value_clip_coef,self.value_clip_coef); value_error=torch.maximum(value_error,(clipped_value-target_return).pow(2))
                    entropy=dist.entropy()
                    losses_for_group=policy+self.value_coef*value_error-self.entropy_coef*entropy; weight=float(len(indices))/float(len(batch)); total_loss=total_loss+losses_for_group.mean()*weight; batch_kl=batch_kl+(((ratio-1.0)-log_ratio).mean().detach()*weight)
                    batch_policy=batch_policy+policy.mean().detach()*weight; batch_value=batch_value+value_error.mean().detach()*weight
                    batch_entropy=batch_entropy+entropy.mean().detach()*weight; batch_clip=batch_clip+((ratio-1.0).abs()>self.clip_coef).float().mean().detach()*weight
                    batch_ratio=batch_ratio+ratio.mean().detach()*weight; ratios_all.append(ratio.detach().cpu().numpy())
                if not bool(torch.isfinite(total_loss)): raise FloatingPointError("PPO loss became non-finite")
                total_loss.backward(); grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm,error_if_nonfinite=True); self.optimizer.step(); losses.append(float(total_loss.item())); measured_kl.append(float(batch_kl.item()))
                policy_losses.append(float(batch_policy.item())); value_losses.append(float(batch_value.item())); entropies.append(float(batch_entropy.item())); clip_fractions.append(float(batch_clip.item())); grad_norms.append(float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm))
                if self.target_kl is not None and measured_kl[-1] > 1.5*self.target_kl: stop_early=True; break
            if stop_early: break
        buffer.clear()
        if self.lr_scheduler is not None: self.lr_scheduler.step()
        concatenated=np.concatenate(ratios_all) if ratios_all else np.zeros(1,np.float32)
        return_variance=float(np.var(rollout_returns))
        return {"loss":float(np.mean(losses)),"learning_rate":self.learning_rate,"approx_kl":float(np.mean(measured_kl)),
                "policy_loss":float(np.mean(policy_losses)),"value_loss":float(np.mean(value_losses)),
                "entropy":float(np.mean(entropies)),"clip_fraction":float(np.mean(clip_fractions)),
                "update_batches":float(len(losses)),"grad_norm":float(np.mean(grad_norms)),
                "train_ratio_mean":float(concatenated.mean()),"train_ratio_max":float(concatenated.max()),
                "explained_variance":explained_variance(rollout_values,rollout_returns) if return_variance > 1e-12 else None,
                # A degenerate (zero-variance) return target makes explained
                # variance undefined; the flag keeps "0.0" from being read as a
                # measured "critic explains nothing" result.
                "explained_variance_defined":bool(return_variance>1e-12),
                "return_variance":return_variance}
