"""独立策略/价值优化、保持原目标单位的 PopArt 和多回合向量 GAE。"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .pareto_agent import vector_gae

VERSION='preference-vector-mappo-conditioned-v44'


def network(inputs,hidden,outputs):
    model=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,outputs))
    for layer in model:
        if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model[-1].weight,.01)
    return model


class ConditionedNetwork(nn.Module):
    """状态与偏好分开编码；输出层接口供PopArt保持原单位。"""
    def __init__(self,state_dim,hidden,outputs,gain=.01):
        super().__init__()
        self.state=nn.Sequential(nn.LayerNorm(state_dim),nn.Linear(state_dim,hidden),nn.Tanh())
        self.preference=nn.Sequential(nn.Linear(3,hidden),nn.Tanh())
        self.fusion=nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh())
        self.output=nn.Linear(hidden,outputs)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.output.weight,gain)
    def forward(self,x):
        return self.output(self.fusion(torch.cat([self.state(x[...,:-3]),self.preference(x[...,-3:])],-1)))
    def __getitem__(self,index):
        if index!=-1:raise IndexError(index)
        return self.output


def balanced_preference(rng,episode):
    # 每四回合保证覆盖三顶点与均衡点；下一组覆盖三个近顶点与随机内部点。
    k=episode%8
    if k<3:return np.eye(3)[k]
    if k==3:return np.ones(3)/3
    if k<7:return .7*np.eye(3)[k-4]+.1
    return rng.dirichlet(np.ones(3))


class ParetoV44(nn.Module):
    def __init__(self,obs_dim,n_agents,hidden=64,lr=3e-4,target_kl=.02,action_mask=None,fixed_preference=(),horizon=24,critic_gain=.01,critic_warmup=0,critic_extra=0):
        super().__init__();self.obs_dim=obs_dim;self.n_agents=n_agents;self.target_kl=target_kl
        self.actor=ConditionedNetwork(obs_dim,hidden,1);self.critic=ConditionedNetwork(obs_dim*n_agents,hidden,3,critic_gain)
        self.register_buffer('action_mask',torch.as_tensor(np.ones(n_agents) if action_mask is None else action_mask,dtype=torch.float32))
        self.register_buffer('fixed_preference',torch.as_tensor(fixed_preference,dtype=torch.float32))
        self.horizon=horizon;self.critic_warmup=critic_warmup;self.critic_extra=critic_extra
        if self.action_mask.shape!=(n_agents,) or self.action_mask.sum()<=0:raise ValueError('有效动作掩码非法')
        self.log_std=nn.Parameter(torch.full((1,),-.5))
        self.register_buffer('return_mean',torch.zeros(3));self.register_buffer('return_std',torch.ones(3))
        self.register_buffer('return_count',torch.tensor(0.,dtype=torch.float64));self.register_buffer('return_m2',torch.zeros(3,dtype=torch.float64))
        self.actor_optimizer=torch.optim.Adam([*self.actor.parameters(),self.log_std],lr=lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

    def state_features(self,obs):
        x=obs.clone()
        # 固定、单调的单位变换，不拟合开发或确认性测试数据。
        x[...,18:36]=torch.asinh(x[...,18:36])
        return x

    def resolved_preference(self,w):
        return self.fixed_preference.expand_as(w) if self.fixed_preference.numel() else w

    def distribution(self,obs,preference):
        w=self.resolved_preference(preference).expand(*obs.shape[:-1],3)
        mean=self.actor(torch.cat([self.state_features(obs),w],-1))
        return Normal(mean,self.log_std.clamp(-5,1).exp().expand_as(mean))

    def normalized_value(self,obs,w):return self.critic(torch.cat([self.state_features(obs).reshape(*obs.shape[:-2],-1),self.resolved_preference(w)],-1))
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
        first_update=bool(self.return_count==0)
        self.update_statistics(target);normalized=(target-self.return_mean)/self.return_std
        mask=self.action_mask[None,:].expand(len(obs),-1)
        def active_mean(x):return (x*mask).sum()/mask.sum()
        actor_params=[*self.actor.parameters(),self.log_std];stopped=False;actor_updates=0
        for _ in range(epochs):
            d=self.distribution(obs,w[:,None,:]);lr=d.log_prob(actions).sum(-1)-old;ratio=lr.exp()
            policy=-active_mean(torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage))
            if not stopped:
                self.actor_optimizer.zero_grad();(policy-entropy*active_mean(d.entropy().squeeze(-1))).backward()
                actor_grad=nn.utils.clip_grad_norm_(actor_params,.5);self.actor_optimizer.step();actor_updates+=1
                with torch.no_grad():
                    lr=self.distribution(obs,w[:,None,:]).log_prob(actions).sum(-1)-old
                    stopped=bool(active_mean((lr.exp()-1)-lr)>self.target_kl)
            value=nn.functional.smooth_l1_loss(self.normalized_value(obs,w),normalized)
            self.critic_optimizer.zero_grad();value.backward();critic_grad=nn.utils.clip_grad_norm_(self.critic.parameters(),.5);self.critic_optimizer.step()
        # actor仍使用采样时的向量GAE；额外拟合仅改善下一批rollout的价值基线。
        extra=max(0,self.critic_warmup-epochs) if first_update else self.critic_extra
        for _ in range(extra):
            value=nn.functional.smooth_l1_loss(self.normalized_value(obs,w),normalized)
            self.critic_optimizer.zero_grad();value.backward();critic_grad=nn.utils.clip_grad_norm_(self.critic.parameters(),.5);self.critic_optimizer.step()
        with torch.no_grad():
            d=self.distribution(obs,w[:,None,:]);lr=d.log_prob(actions).sum(-1)-old;mse=((self.value(obs,w)-target)**2).mean(0)
            metrics=dict(policy_loss=float(policy),value_loss=float(value),approx_kl=float(active_mean((lr.exp()-1)-lr)),
                clip_fraction=float(active_mean(((lr.exp()-1).abs()>clip).float())),entropy=float(active_mean(d.entropy().squeeze(-1))),
                actor_grad_norm=float(actor_grad),critic_grad_norm=float(critic_grad),actor_updates=actor_updates,critic_updates=epochs+extra,actor_early_stop=stopped,
                action_saturation=float(active_mean((actions.tanh().abs()>.95).float().squeeze(-1))))
            metrics.update({f'value_mse_{k}':float(mse[i]) for i,k in enumerate(('cost','carbon','reserve'))})
        with torch.no_grad():
            prediction=self.value(obs,w);variance=target.var(0,unbiased=False)
            explained=1-(target-prediction).var(0,unbiased=False)/(variance+1e-8)
            metrics.update({f'value_explained_variance_{k}':float(explained[i]) for i,k in enumerate(('cost','carbon','reserve'))})
            metrics.update({f'adv_std_{k}':float(np.std(adv[:,i])) for i,k in enumerate(('cost','carbon','reserve'))})
            probe=torch.eye(3);means=torch.stack([self.distribution(obs,p).mean for p in probe])
            metrics['preference_action_span']=float(active_mean((means.max(0).values-means.min(0).values).squeeze(-1)))
            metrics['active_actions']=float(self.action_mask.sum())
        if not all(np.isfinite(v) for v in metrics.values()):raise RuntimeError('v4更新出现非有限数')
        return metrics
