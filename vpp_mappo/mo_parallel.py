"""借鉴 MOMAland 的并行向量奖励接口；内部普通 MAPPO 入口独立保留。"""
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from .environment import VPPAdapter
from .objectives import OBJECTIVE_NAMES


class VPPMOParallelEnv(ParallelEnv):
    metadata = {'name':'vpp_ieee33_mo_v0','render_modes':[],'is_parallelizable':True}
    render_mode = None

    def __init__(self, config, csv_path=None):
        if not config.metrics_enabled: raise ValueError('向量接口必须启用三目标记录')
        self.metadata = dict(type(self).metadata, name=f'vpp_{config.network_model}_mo_v0')
        self.core = VPPAdapter(config,csv_path)
        self.possible_agents = getattr(self.core,'agent_names',['ess','ev','dr'])
        self.agents = []
        self.observation_spaces = {a:self.core.observation_space for a in self.possible_agents}
        self.action_spaces = {a:self.core.action_space for a in self.possible_agents}
        self.reward_spaces = {a:Box(-np.inf,np.inf,(3,),dtype=np.float32) for a in self.possible_agents}
        self.state_space = self.core.share_observation_space
        self._state = None

    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]
    def reward_space(self, agent): return self.reward_spaces[agent]

    def reset(self, seed=None, options=None):
        seed = self.core.config.seed if seed is None else seed
        index = (options or {}).get('profile_index',0)
        obs,shared = self.core.reset(seed,index)
        self.agents = self.possible_agents.copy(); self._state = shared[0].copy()
        return {a:obs[i].copy() for i,a in enumerate(self.agents)}, {a:{} for a in self.agents}

    def state(self):
        if self._state is None: raise RuntimeError('请先 reset')
        return self._state.copy()

    def step(self, actions):
        if not self.agents:
            if actions: raise RuntimeError('终止后不能继续提交动作')
            return {},{},{},{},{}
        if set(actions)!=set(self.agents): raise ValueError('动作必须完整覆盖当前智能体')
        obs,shared,_,done,info = self.core.step(np.stack([actions[a] for a in self.agents]))
        self._state = shared[0].copy()
        vector=np.asarray([info[k] for k in OBJECTIVE_NAMES],dtype=np.float32)
        # 团队目标均分，外部算法按智能体求和时不会重复累计三遍。
        rewards={a:vector.copy()/len(self.agents) for a in self.agents}
        observations={a:obs[i].copy() for i,a in enumerate(self.agents)}
        terminations={a:done for a in self.agents}; truncations={a:False for a in self.agents}
        infos={a:dict(info,team_vector=vector.tolist(),safety_penalty=self.core.config.violation_penalty*info['constraint_violations']/self.core.config.reward_scale) for a in self.agents}
        if done: self.agents=[]
        return observations,rewards,terminations,truncations,infos

    def close(self):
        self.agents=[]
