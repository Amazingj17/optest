"""Dependency-free dense graph attention actor-critic for resource selection."""
from __future__ import annotations
import torch
from torch import nn

class DenseGATLayer(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, heads:int=1, edge_dim:int=2) -> None:
        super().__init__(); self.heads=heads; self.proj=nn.Linear(input_dim,output_dim*heads,bias=False); self.attn=nn.Parameter(torch.empty(heads,2*output_dim)); self.edge_attention=nn.Linear(edge_dim,heads,bias=False); nn.init.xavier_uniform_(self.attn)
    def forward(self,x:torch.Tensor,adj:torch.Tensor,edge_features:torch.Tensor)->torch.Tensor:
        b,n,_=x.shape; h=self.proj(x).view(b,n,self.heads,-1); left=(h*self.attn[:,:h.shape[-1]]).sum(-1); right=(h*self.attn[:,h.shape[-1]:]).sum(-1); score=torch.nn.functional.leaky_relu(left.unsqueeze(2)+right.unsqueeze(1),.2).permute(0,3,1,2); edge_score=self.edge_attention(edge_features).permute(0,3,1,2); score=(score+edge_score).masked_fill(~adj.unsqueeze(1),-1e9); alpha=torch.softmax(score,-1); out=torch.einsum('bhij,bjhd->bihd',alpha,h).mean(2); return torch.nn.functional.elu(out)

class LowLevelGATActorCritic(nn.Module):
    def __init__(self, node_dim:int, task_dim:int, hidden_dim:int=64, heads:int=4, use_gat:bool=True, heuristic_residual:bool=False, heuristic_weight:float=8.0, residual_limit:float=0.5) -> None:
        super().__init__(); self.use_gat=bool(use_gat); self.heuristic_residual=bool(heuristic_residual); self.heuristic_weight=float(heuristic_weight); self.residual_limit=float(residual_limit); self.gat1=DenseGATLayer(node_dim,hidden_dim,heads) if self.use_gat else None; self.gat2=DenseGATLayer(hidden_dim,hidden_dim) if self.use_gat else None; self.node_encoder=nn.Linear(node_dim,hidden_dim); self.task=nn.Linear(task_dim,hidden_dim); self.actor=nn.Sequential(nn.Linear(hidden_dim*2+node_dim-1,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1)); self.critic=nn.Sequential(nn.Linear(hidden_dim+task_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
        if self.heuristic_residual:
            if node_dim <= 5: raise ValueError("HEFT residual low actor requires normalized EFT feature 5")
            if self.heuristic_weight <= 0.0 or self.residual_limit < 0.0: raise ValueError("invalid low heuristic residual scales")
            nn.init.zeros_(self.actor[-1].weight); nn.init.zeros_(self.actor[-1].bias)
    def forward(self,node_features:torch.Tensor,task_features:torch.Tensor,adjacency:torch.Tensor,edge_features:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor]:
        if edge_features is None: edge_features=torch.zeros((*adjacency.shape,2),device=node_features.device,dtype=node_features.dtype)
        h=torch.tanh(self.node_encoder(node_features)) if not self.use_gat else self.gat2(self.gat1(node_features,adjacency,edge_features),adjacency,edge_features); t=torch.tanh(self.task(task_features)); timing=node_features[...,1:]; actor_logits=self.actor(torch.cat((h,t.unsqueeze(1).expand_as(h),timing),-1)).squeeze(-1); logits=actor_logits
        if self.heuristic_residual:
            logits=-self.heuristic_weight*node_features[...,5]+self.residual_limit*torch.tanh(actor_logits)
        value=self.critic(torch.cat((h.mean(1),task_features),-1)).squeeze(-1); return logits,value
