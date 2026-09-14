"""三资源智能体适配器；通信和计算配置固定，作为第一版 MAPPO 基线。"""
from typing import Protocol
import numpy as np
from gymnasium.spaces import Box
from vpp.dt_c3.environment import PowerIoTVPPEnvironment, VPPEnvConfig
from vpp.dt_c3.communication import CommunicationConfig
from .data import CSVProfiles


class MultiAgentEnvironment(Protocol):
    num_agents: int
    data: CSVProfiles | None
    observation_space: Box
    share_observation_space: Box
    action_space: Box

    def reset(self, scenario_seed: int, profile_index: int = 0): ...
    def step(self, actions): ...


class VPPAdapter:
    num_agents = 3
    observation_space = Box(-np.inf, np.inf, (10,), dtype=np.float32)
    share_observation_space = Box(-np.inf, np.inf, (30,), dtype=np.float32)
    action_space = Box(-np.inf, np.inf, (1,), dtype=np.float32)

    def __init__(self, config, csv_path=None):
        self.config = config
        self.data = CSVProfiles(csv_path, config.horizon) if csv_path else None
        self.env = None

    def reset(self, scenario_seed, profile_index=0):
        cfg = self.config
        channel = CommunicationConfig(seed=int(scenario_seed), fixed_delay_steps=cfg.delay_steps,
                                      max_delay_steps=cfg.delay_steps, packet_loss_rate=cfg.packet_loss,
                                      queue_delay_mean_steps=0, command_downlink_delay_steps=0)
        self.env = PowerIoTVPPEnvironment(VPPEnvConfig(
            horizon_steps=cfg.horizon, dt_hours=cfg.dt_hours, seed=int(scenario_seed),
            communication=channel, use_digital_twin=cfg.digital_twin, use_grid_constraints=False))
        if self.data:
            self.env.profiles = self.data.get(profile_index)
        state = self.env.reset()
        return self.encode(state)

    def encode(self, state):
        # 公共广播：时间、电价和聚合净负荷。每个资源只读取自己的私有状态。
        cfg = self.config
        net = (state['load_kw'] - state['pv_kw'] - state['wind_kw']) / 1000
        common = [state['step'] / cfg.horizon, state['price'] / 0.3, net,
                  state['avg_aoi'] / cfg.horizon, state['packet_loss_ratio']]
        own = [(state['ess_soc'], 0.55), (state['ev_soc'], 0.55), (state['load_kw'] / 1000, 0.0)]
        obs = np.array([common + list(own[i]) + np.eye(3)[i].tolist() for i in range(3)], dtype=np.float32)
        share = np.repeat(obs.reshape(1, -1), 3, axis=0)
        return obs, share

    def step(self, actions):
        raw = np.asarray(actions, dtype=np.float32)
        if raw.shape != (3, 1) or not np.isfinite(raw).all():
            raise ValueError('动作必须为有限的 (3, 1) 数组')
        # 缓存保留高斯原始动作及其概率；环境中的 tanh 和安全映射是确定性转移。
        bounded = np.tanh(raw[:, 0])
        action = dict(ess_power_kw=float(250 * bounded[0]), ev_power_kw=float(300 * bounded[1]),
                      dr_kw=float(75 * (bounded[2] + 1)), upload_priority=1.0,
                      edge_cpu_fraction=1.0, offload_ratio=0.0)
        price = self.env.true_state['price']
        result = self.env.step(action, apply_safety=self.config.safety)
        info = result.info
        applied = info['applied_action']
        dt = self.config.dt_hours
        energy = info['grid_power_kw'] * dt
        # 第一版使用纯经济费用与独立罚项，排除机器墙钟时间和旧 C3 人工费用。
        cost = (max(0, energy) * price - max(0, -energy) * price * 0.65
                + 0.01 * (abs(applied['ess_power_kw']) + abs(applied['ev_power_kw'])) * dt
                + 0.03 * applied['dr_kw'] * dt + 0.05 * info['curtailment_kw'] * dt)
        terminal_penalty = 0.0
        if result.done:
            terminal_penalty = self.config.terminal_soc_penalty * sum(
                max(0, 0.55 - info['post_state'][key]) for key in ('ess_soc', 'ev_soc'))
        reward = -(cost + self.config.violation_penalty * info['constraint_violations'] + terminal_penalty) / self.config.reward_scale
        clean = dict(cost=float(cost), reward=float(reward), constraint_violations=info['constraint_violations'],
                     pre_shield_violations=info['pre_shield_violations'],
                     ess_soc=info['post_state']['ess_soc'], ev_soc=info['post_state']['ev_soc'],
                     grid_power_kw=info['grid_power_kw'], terminal_penalty=terminal_penalty,
                     ess_power_kw=applied['ess_power_kw'], ev_power_kw=applied['ev_power_kw'], dr_kw=applied['dr_kw'])
        obs, share = (np.zeros((3, 10), np.float32), np.zeros((3, 30), np.float32)) if result.done else self.encode(result.state)
        # horizon 是含终端成本的真实任务终点，不是训练时人为截断。
        return obs, share, np.full((3, 1), reward, np.float32), result.done, clean
