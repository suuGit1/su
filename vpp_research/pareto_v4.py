"""独立策略/价值优化、保持原目标单位的 PopArt 和多回合向量 GAE。"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .pareto_agent import vector_gae

VERSION='preference-vector-mappo-popart-v4'


def network(inputs,hidden,outputs):
    model=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,outputs))
    for layer in model:
        if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model[-1].weight,.01)
    return model


class ParetoV4(nn.Module):
    def __init__(self,obs_dim,n_agents,hidden=64,lr=3e-4,target_kl=.02):
        super().__init__();self.obs_dim=obs_dim;self.n_agents=n_agents;self.target_kl=target_kl
        self.actor=network(obs_dim+3,hidden,1);self.critic=network(obs_dim*n_agents+3,hidden,3)
        self.log_std=nn.Parameter(torch.full((1,),-.5))
        self.register_buffer('return_mean',torch.zeros(3));self.register_buffer('return_std',torch.ones(3))
        self.register_buffer('return_count',torch.tensor(0.,dtype=torch.float64));self.register_buffer('return_m2',torch.zeros(3,dtype=torch.float64))
        self.actor_optimizer=torch.optim.Adam([*self.actor.parameters(),self.log_std],lr=lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

    def distribution(self,obs,preference):
        w=preference.expand(*obs.shape[:-1],3);mean=self.actor(torch.cat([obs,w],-1))
        return Normal(mean,self.log_std.clamp(-5,1).exp().expand_as(mean))

    def normalized_value(self,obs,w):return self.critic(torch.cat([obs.reshape(*obs.shape[:-2],-1),w],-1))
    def value(self,obs,w):return self.normalized_value(obs,w)*self.return_std+self.return_mean

    def act(self,obs,preference,deterministic=False):
        o=torch.as_tensor(obs,dtype=torch.float32);w=torch.as_tensor(preference,dtype=torch.float32)
        with torch.no_grad():
            d=self.distribution(o,w);a=d.mean if deterministic else d.sample()
            return a.numpy(),d.log_prob(a).sum(-1).numpy(),self.value(o,w).numpy()

    @torch.no_grad()
    def update_statistics(self,targets):
        old_mean=self.return_mean.clone();old_std=self.return_std.clone()
        n=len(targets);mean=targets.double().mean(0);m2=((targets.double()-mean)**2).sum(0)
        total=self.return_count+n;delta=mean-old_mean.double();new_mean=old_mean.double()+delta*n/total
        new_m2=self.return_m2+m2+delta.square()*self.return_count*n/total
        new_std=(new_m2/total).clamp(min=1e-4).sqrt().float();last=self.critic[-1]
        last.weight.mul_((old_std/new_std)[:,None]);last.bias.copy_((old_std*last.bias+old_mean-new_mean.float())/new_std)
        self.return_mean.copy_(new_mean);self.return_std.copy_(new_std);self.return_m2.copy_(new_m2);self.return_count.copy_(total)
        # 输出层坐标变化后清除其Adam动量，避免混用归一化前后的单位。
        self.critic_optimizer.state.pop(last.weight,None);self.critic_optimizer.state.pop(last.bias,None)

    def optimizer_state(self):return dict(actor=self.actor_optimizer.state_dict(),critic=self.critic_optimizer.state_dict())
    def load_optimizer_state(self,state):
        self.actor_optimizer.load_state_dict(state['actor']);self.critic_optimizer.load_state_dict(state['critic'])

    def update(self,rollout,epochs=5,clip=.2,gamma=.99,lam=.95,entropy=.01):
        obs=torch.as_tensor(np.asarray(rollout['obs']),dtype=torch.float32);w=torch.as_tensor(np.asarray(rollout['preferences']),dtype=torch.float32)
        actions=torch.as_tensor(np.asarray(rollout['actions']),dtype=torch.float32);old=torch.as_tensor(np.asarray(rollout['logprobs']),dtype=torch.float32)
        values=np.r_[np.asarray(rollout['values']),np.zeros((1,3))]
        adv,returns=vector_gae(rollout['vectors'],values,rollout['dones'],gamma,lam)
        # critic归一化不改变策略所使用的经济—碳—备用权重和原始量纲。
        scalar=np.sum(adv*np.asarray(rollout['preferences']),axis=1);scalar=(scalar-scalar.mean())/(scalar.std()+1e-8)
        advantage=torch.as_tensor(scalar,dtype=torch.float32)[:,None];target=torch.as_tensor(returns,dtype=torch.float32)
        self.update_statistics(target);normalized=(target-self.return_mean)/self.return_std
        actor_params=[*self.actor.parameters(),self.log_std];stopped=False;actor_updates=0
        for _ in range(epochs):
            d=self.distribution(obs,w[:,None,:]);lr=d.log_prob(actions).sum(-1)-old;ratio=lr.exp()
            policy=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
            if not stopped:
                self.actor_optimizer.zero_grad();(policy-entropy*d.entropy().mean()).backward()
                actor_grad=nn.utils.clip_grad_norm_(actor_params,.5);self.actor_optimizer.step();actor_updates+=1
                with torch.no_grad():
                    lr=self.distribution(obs,w[:,None,:]).log_prob(actions).sum(-1)-old
                    stopped=bool(((lr.exp()-1)-lr).mean()>self.target_kl)
            value=nn.functional.smooth_l1_loss(self.normalized_value(obs,w),normalized)
            self.critic_optimizer.zero_grad();value.backward();critic_grad=nn.utils.clip_grad_norm_(self.critic.parameters(),.5);self.critic_optimizer.step()
        with torch.no_grad():
            d=self.distribution(obs,w[:,None,:]);lr=d.log_prob(actions).sum(-1)-old;mse=((self.value(obs,w)-target)**2).mean(0)
            metrics=dict(policy_loss=float(policy),value_loss=float(value),approx_kl=float(((lr.exp()-1)-lr).mean()),
                clip_fraction=float(((lr.exp()-1).abs()>clip).float().mean()),entropy=float(d.entropy().mean()),
                actor_grad_norm=float(actor_grad),critic_grad_norm=float(critic_grad),actor_updates=actor_updates,actor_early_stop=stopped,
                action_saturation=float((actions.tanh().abs()>.95).float().mean()))
            metrics.update({f'value_mse_{k}':float(mse[i]) for i,k in enumerate(('cost','carbon','reserve'))})
        if not all(np.isfinite(v) for v in metrics.values()):raise RuntimeError('v4更新出现非有限数')
        return metrics
