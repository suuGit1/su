"""偏好条件高斯 actor、集中式三维 critic 与向量 GAE；独立于官方普通 MAPPO。"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

VERSION='preference-vector-mappo-v1'


def vector_gae(rewards,values,dones,gamma=.99,lam=.95):
    r=np.asarray(rewards,dtype=np.float32);v=np.asarray(values,dtype=np.float32);d=np.asarray(dones,dtype=np.float32)
    if r.ndim!=2 or r.shape[1]!=3 or v.shape!=(len(r)+1,3) or d.shape!=(len(r),):raise ValueError('向量 GAE 维度错误')
    adv=np.zeros_like(r);carry=np.zeros(3,dtype=np.float32)
    for t in reversed(range(len(r))):
        mask=1-d[t];delta=r[t]+gamma*v[t+1]*mask-v[t]
        carry=delta+gamma*lam*mask*carry;adv[t]=carry
    return adv,adv+v[:-1]


class ParetoAgent(nn.Module):
    def __init__(self,obs_dim,n_agents,hidden=32,lr=3e-4):
        super().__init__();self.obs_dim=obs_dim;self.n_agents=n_agents
        def mlp(n,k):return nn.Sequential(nn.Linear(n,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,k))
        self.actor=mlp(obs_dim+3,1);self.critic=mlp(obs_dim*n_agents+3,3);self.log_std=nn.Parameter(torch.zeros(1))
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)

    def distribution(self,obs,preference):
        w=preference.expand(*obs.shape[:-1],3)
        mean=self.actor(torch.cat([obs,w],dim=-1))
        return Normal(mean,self.log_std.clamp(-5,2).exp().expand_as(mean))

    def value(self,obs,preference):return self.critic(torch.cat([obs.reshape(*obs.shape[:-2],-1),preference],dim=-1))

    def act(self,obs,preference,deterministic=False):
        o=torch.as_tensor(obs,dtype=torch.float32);w=torch.as_tensor(preference,dtype=torch.float32)
        with torch.no_grad():
            dist=self.distribution(o,w);a=dist.mean if deterministic else dist.sample()
            return a.numpy(),dist.log_prob(a).sum(-1).numpy(),self.value(o,w).numpy()

    def update(self,rollout,epochs=2,clip=.2,gamma=.99,lam=.95,entropy=.01):
        obs=torch.as_tensor(np.array(rollout['obs']),dtype=torch.float32)
        w=torch.as_tensor(np.array(rollout['preferences']),dtype=torch.float32)
        actions=torch.as_tensor(np.array(rollout['actions']),dtype=torch.float32)
        old=torch.as_tensor(np.array(rollout['logprobs']),dtype=torch.float32)
        values=np.r_[np.array(rollout['values']),np.zeros((1,3))]
        adv,returns=vector_gae(rollout['vectors'],values,rollout['dones'],gamma,lam)
        # 先保留向量时序优势，再按每回合固定偏好标量化；不单独标准化每个目标以免改变偏好。
        scalar=np.sum(adv*np.array(rollout['preferences']),axis=1)
        scalar=(scalar-scalar.mean())/(scalar.std()+1e-8);a=torch.as_tensor(scalar,dtype=torch.float32)[:,None]
        target=torch.as_tensor(returns,dtype=torch.float32)
        for _ in range(epochs):
            dist=self.distribution(obs,w[:,None,:]);logp=dist.log_prob(actions).sum(-1);ratio=(logp-old).exp()
            policy=-torch.minimum(ratio*a,ratio.clamp(1-clip,1+clip)*a).mean()
            value=((self.value(obs,w)-target)**2).mean()
            loss=policy+.5*value-entropy*dist.entropy().mean()
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.parameters(),.5);self.optimizer.step()
        if not torch.isfinite(loss):raise RuntimeError('Pareto 更新损失非有限')
        return dict(policy_loss=float(policy.detach()),value_loss=float(value.detach()))


def sample_preference(rng,episode):
    # 顶点与内部连续偏好混合；偏好在一个完整回合中固定。
    return np.eye(3)[episode%3] if episode%4==0 else rng.dirichlet(np.ones(3))
