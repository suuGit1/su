"""算法注册：MAPPO 与 IPPO 共用官方 PPO 更新器，价值函数信息不同。"""
from dataclasses import asdict
import numpy as np
import torch
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.shared_buffer import SharedReplayBuffer

UPSTREAM_COMMIT = 'de66d7a4b23fac2513f56f96f73b3f5cb96695ac'
REGISTRY = {'mappo': True, 'ippo': False}


class OfficialPPO:
    def __init__(self, config, env):
        args = get_config().parse_args([])
        args.algorithm_name = 'mappo'
        args.episode_length = config.horizon
        args.n_rollout_threads = 1
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
        args.use_popart = False
        args.use_valuenorm = True
        args.use_proper_time_limits = True
        args.recurrent_N = 1
        for k in ('hidden_size', 'ppo_epoch', 'num_mini_batch', 'lr', 'gamma', 'gae_lambda', 'clip_param', 'entropy_coef'):
            setattr(args, k, getattr(config, k))
        args.critic_lr = config.lr
        self.args = args
        self.centralized = REGISTRY[config.algorithm]
        args.use_centralized_V = self.centralized
        critic_space = env.share_observation_space if self.centralized else env.observation_space
        self.policy = R_MAPPOPolicy(args, env.observation_space, critic_space, env.action_space,
                                   device=torch.device(config.device))
        self.trainer = R_MAPPO(args, self.policy, device=torch.device(config.device))
        self.env = env

    def critic_obs(self, obs, share):
        return share if self.centralized else obs

    def buffer(self):
        critic_space = self.env.share_observation_space if self.centralized else self.env.observation_space
        return SharedReplayBuffer(self.args, self.env.num_agents, self.env.observation_space,
                                  critic_space, self.env.action_space)

    def state(self):
        return dict(actor=self.policy.actor.state_dict(), critic=self.policy.critic.state_dict(),
                    actor_optimizer=self.policy.actor_optimizer.state_dict(),
                    critic_optimizer=self.policy.critic_optimizer.state_dict(),
                    value_normalizer=self.trainer.value_normalizer.state_dict())

    def load(self, state):
        self.policy.actor.load_state_dict(state['actor'])
        self.policy.critic.load_state_dict(state['critic'])
        self.trainer.value_normalizer.load_state_dict(state['value_normalizer'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
