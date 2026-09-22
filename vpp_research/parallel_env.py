"""MOMAland 风格的向量奖励 Parallel 接口；不改变现有训练器。

独立按公开接口实现，未依赖或声称通过 MOMAland 的上游兼容性认证。
每个智能体收到相同团队奖励，团队回报只能计一次，不能按智能体求和。
"""
import copy
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from .environment import ResearchEnv, OBS_VERSION, SCALES


class VPPParallelEnv(ParallelEnv):
    metadata = {'name': 'c3_vpp_vector_v1', 'render_modes': [], 'is_parallelizable': True}
    render_mode = None

    def __init__(self, config, **kwargs):
        if kwargs.get('vector_metrics') is False:
            raise ValueError('向量接口要求 vector_metrics=True')
        self.env = ResearchEnv(config, **kwargs)
        self.possible_agents = list(self.env.agent_names)
        self.agents = []
        self.observation_spaces = {a: Box(-np.inf, np.inf, (65,), np.float32) for a in self.possible_agents}
        # 原始高斯动作；ResearchEnv 内部统一执行 tanh，不能再次压缩。
        self.action_spaces = {a: Box(-np.inf, np.inf, (1,), np.float32) for a in self.possible_agents}
        self.reward_spaces = {a: Box(-np.inf, np.inf, (3,), np.float64) for a in self.possible_agents}
        self.state_space = Box(-np.inf, np.inf, (65 * len(self.possible_agents),), np.float32)
        self._obs = None
        self.np_random, _ = seeding.np_random()

    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]
    def reward_space(self, agent): return self.reward_spaces[agent]

    def _observations(self):
        return {a: self._obs[i].copy() for i, a in enumerate(self.possible_agents)}

    def reset(self, seed=None, options=None):
        options = {} if options is None else options
        if set(options) - {'profile_index'}:
            raise ValueError('未知 reset options')
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        scenario_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
        self._obs, _ = self.env.reset(scenario_seed, options.get('profile_index', 0))
        self.agents = self.possible_agents.copy()
        for i, a in enumerate(self.agents):
            self.action_space(a).seed(scenario_seed + i)
        info = dict(observation_version=OBS_VERSION, objective_version=self.env.objective_version,
                    objective_scales=SCALES.tolist(), team_reward=True, scenario_seed=scenario_seed)
        return self._observations(), {a: copy.deepcopy(info) for a in self.agents}

    def state(self):
        if self._obs is None:
            raise RuntimeError('请先 reset')
        # 仅缓存的公开观察；不访问真实物理状态。
        return self._obs.reshape(-1).copy()

    def step(self, actions):
        if not self.agents:
            raise RuntimeError('回合未开始或已经结束，请 reset')
        if set(actions) != set(self.agents):
            raise ValueError('动作必须恰好覆盖所有活动智能体')
        raw = np.asarray([actions[a] for a in self.possible_agents], dtype=float)
        if raw.shape != (len(self.agents), 1) or not np.isfinite(raw).all():
            raise ValueError('动作必须为有限的一维数组')
        self._obs, _, _, done, info = self.env.step(raw)
        vector = np.asarray(info['objective_vector'], dtype=float)
        penalty = self.env.config.violation_penalty * info['constraint_violations'] / self.env.config.reward_scale
        rewards = {a: (vector - penalty).copy() for a in self.agents}
        info = dict(info, raw_objective_vector=vector.tolist(), vector_penalty=float(penalty), team_reward=True)
        infos = {a: copy.deepcopy(info) for a in self.agents}
        terminations = {a: bool(done) for a in self.agents}
        truncations = {a: False for a in self.agents}
        observations = self._observations()
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.agents = []

    def render(self):
        raise NotImplementedError('此研究接口不提供渲染')
