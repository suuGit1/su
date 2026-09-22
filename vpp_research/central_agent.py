"""集中式 PPO：联合公开观察、联合动作概率和经济标量价值。"""
import torch
from torch import nn
from torch.distributions import Normal
from .pareto_agent import ParetoAgent


class CentralPPO(ParetoAgent):
    def __init__(self,obs_dim,n_agents,hidden=32,lr=3e-4):
        super().__init__(obs_dim,n_agents,hidden,lr)
        def mlp(n,k):return nn.Sequential(nn.Linear(n,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,k))
        self.actor=mlp(obs_dim*n_agents,n_agents)
        self.critic=mlp(obs_dim*n_agents,1)
        self.log_std=nn.Parameter(torch.zeros(n_agents))
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)

    def distribution(self,obs,preference):
        mean=self.actor(obs.reshape(*obs.shape[:-2],-1))
        base=Normal(mean,self.log_std.clamp(-5,2).exp().expand_as(mean))
        return JointDistribution(base)

    def value(self,obs,preference):
        # 复用向量缓存时复制同一个经济价值；回报也复制经济分量，等价于标量 MSE。
        return self.critic(obs.reshape(*obs.shape[:-2],-1)).expand(*obs.shape[:-2],3)


class JointDistribution:
    def __init__(self,base):self.base=base
    @property
    def mean(self):return self.base.mean.unsqueeze(-1)
    def sample(self):return self.base.sample().unsqueeze(-1)
    def log_prob(self,actions):
        joint=self.base.log_prob(actions.squeeze(-1)).sum(-1,keepdim=True)
        return joint.unsqueeze(-1).expand(*actions.shape)
    def entropy(self):return self.base.entropy().sum(-1)
