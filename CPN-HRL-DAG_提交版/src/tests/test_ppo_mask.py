from __future__ import annotations
import numpy as np
import torch
from cpn_hrl_dag.algorithms.ppo import PPOAgent,RolloutBuffer
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic

def test_masked_ppo_never_samples_illegal_action() -> None:
    model=HighLevelLSTMActorCritic(3,2,8); device=torch.device("cpu")
    def tensor(obs): return (torch.tensor(obs["task_features"]).unsqueeze(0),torch.tensor(obs["task_mask"]).unsqueeze(0),torch.tensor(obs["resource_features"]).unsqueeze(0))
    agent=PPOAgent(model,torch.optim.Adam(model.parameters(),lr=.001),tensor,model.forward,device,.99,.95,.2,.01,batch_size=17)
    obs={"task_features":np.ones((3,3),np.float32),"task_mask":np.ones(3,bool),"resource_features":np.ones((2,2),np.float32)}; mask=np.array([False,True,False])
    buffer=RolloutBuffer()
    for i in range(200):
        action,logprob,value=agent.act(obs,mask); assert action==1; buffer.add(observation=obs,mask=mask,action=action,log_prob=logprob,value=value,reward=-.1,done=i==199)
    agent.update(buffer,epochs=1)


def test_ppo_batches_equal_shapes_and_handles_variable_dag_sizes() -> None:
    model=HighLevelLSTMActorCritic(3,2,8); device=torch.device("cpu")
    def tensor(obs): return (torch.tensor(obs["task_features"]).unsqueeze(0),torch.tensor(obs["task_mask"]).unsqueeze(0),torch.tensor(obs["resource_features"]).unsqueeze(0))
    agent=PPOAgent(model,torch.optim.Adam(model.parameters(),lr=.001),tensor,model.forward,device,1.0,.95,.1,.001,batch_size=8,target_kl=.1,value_clip_coef=.2)
    buffer=RolloutBuffer()
    for tasks in (3,4,3,4):
        obs={"task_features":np.ones((tasks,3),np.float32),"task_mask":np.ones(tasks,bool),"resource_features":np.ones((2,2),np.float32)}; mask=np.ones(tasks,bool); action,logprob,value=agent.act(obs,mask); buffer.add(observation=obs,mask=mask,action=action,log_prob=logprob,value=value,reward=-.1,done=True)
    metrics=agent.update(buffer,epochs=1)
    assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["approx_kl"])
