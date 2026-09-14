"""PyTorch RL algorithms for DT-C3 safe multi-agent energy management.

This module adds paper-grade algorithm components around the lightweight VPP
simulator:

* neural Actor-Critic policies;
* MAPPO-style centralized training / decentralized execution (CTDE);
* independent PPO / IAC baselines;
* MADDPG-style off-policy deterministic actor-critic baseline;
* trajectory and replay buffers;
* actor/critic losses and convergence traces;
* multi-seed experiment utilities.

The implementation is intentionally compact.  It is suitable for reproducible
research experiments and can be extended to a larger MAPPO/MADDPG stack later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
import math
import random
import time

import numpy as np

try:  # pragma: no cover - tested by smoke tests only when torch is available
    import torch
    from torch import nn
    from torch.distributions import Normal
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Normal = None
    F = None

from .environment import PowerIoTVPPEnvironment
from .safety import SafetyShield

STATE_KEYS = [
    "load_kw",
    "pv_kw",
    "wind_kw",
    "ess_soc",
    "ev_soc",
    "price",
    "avg_aoi",
    "packet_loss_ratio",
    "step",
]
ACTION_KEYS = ["ess_power_kw", "ev_power_kw", "dr_kw", "upload_priority", "edge_cpu_fraction", "offload_ratio"]
AGENT_NAMES = ["ess_agent", "ev_agent", "dr_agent", "communication_agent", "computation_agent", "offloading_agent"]


def torch_available() -> bool:
    return torch is not None


def resolve_torch_device(requested: str = "cpu") -> str:
    """Return a usable torch device string.

    `requested="cuda"` uses the first CUDA device when available.  When CUDA
    is requested on a CPU-only machine, the code falls back to CPU so smoke
    tests remain runnable on any computer; on a CUDA-enabled workstation it will
    use the GPU.
    """
    if torch is None:
        return "cpu"
    dev = (requested or "cpu").strip().lower()
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        if torch.cuda.is_available():
            return dev
        print(f"[DT-C3] Requested device '{requested}' but CUDA is not available; falling back to CPU.")
        return "cpu"
    return "cpu"


def describe_torch_device(device: str) -> str:
    if torch is None:
        return "PyTorch unavailable"
    resolved = resolve_torch_device(device)
    if resolved.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in resolved:
            try:
                idx = int(resolved.split(":", 1)[1])
            except ValueError:
                idx = 0
        name = torch.cuda.get_device_name(idx)
        return f"{resolved} ({name})"
    return "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.set_num_threads(1)


@dataclass
class RLTrainConfig:
    """Common training configuration for paper experiments."""

    episodes: int = 100
    horizon_steps: int | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_ratio: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    update_epochs: int = 4
    batch_size: int = 256
    hidden_dim: int = 64
    seed: int = 42
    device: str = "cpu"
    exploration_std: float = 0.35
    safety_enabled: bool = True
    communication_aware: bool = True
    use_aoi: bool = True
    save_checkpoints: bool = True


@dataclass
class RLTrainingTrace:
    """Training curves and loss traces reported in the paper."""

    method: str
    seed: int
    episode_rewards: list[float] = field(default_factory=list)
    episode_costs: list[float] = field(default_factory=list)
    episode_violations: list[int] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    update_times_s: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {
            "train_reward_last": float(self.episode_rewards[-1]) if self.episode_rewards else 0.0,
            "train_reward_mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "actor_loss_last": float(self.actor_losses[-1]) if self.actor_losses else 0.0,
            "critic_loss_last": float(self.critic_losses[-1]) if self.critic_losses else 0.0,
            "train_violations_mean": float(np.mean(self.episode_violations)) if self.episode_violations else 0.0,
        }


class FeatureEncoder:
    """Fixed normalizer for VPP states.

    Fixed scaling is deliberate: it keeps multi-seed runs deterministic and
    avoids leaking evaluation data into normalization statistics.
    """

    scale = {
        "load_kw": 1000.0,
        "pv_kw": 600.0,
        "wind_kw": 300.0,
        "ess_soc": 1.0,
        "ev_soc": 1.0,
        "price": 0.30,
        "avg_aoi": 10.0,
        "packet_loss_ratio": 1.0,
        "step": 96.0,
    }

    def __init__(self, communication_aware: bool = True, use_aoi: bool = True) -> None:
        self.communication_aware = communication_aware
        self.use_aoi = use_aoi

    @property
    def dim(self) -> int:
        return len(STATE_KEYS)

    def encode(self, state: dict[str, float]) -> np.ndarray:
        arr = []
        for k in STATE_KEYS:
            value = float(state.get(k, 0.0))
            if not self.communication_aware and k in {"avg_aoi", "packet_loss_ratio"}:
                value = 0.0
            if not self.use_aoi and k == "avg_aoi":
                value = 0.0
            arr.append(value / self.scale[k])
        return np.asarray(arr, dtype=np.float32)

    def agent_obs(self, state: dict[str, float], agent_index: int) -> np.ndarray:
        one_hot = np.zeros(len(AGENT_NAMES), dtype=np.float32)
        one_hot[agent_index] = 1.0
        return np.concatenate([self.encode(state), one_hot], dtype=np.float32)


class ActionMapper:
    """Map normalized [-1, 1] neural actions to VPP kW actions."""

    def __init__(self, ess_max: float = 250.0, ev_max: float = 300.0, dr_max: float = 150.0) -> None:
        self.ess_max = ess_max
        self.ev_max = ev_max
        self.dr_max = dr_max

    @property
    def dim(self) -> int:
        return 6

    def to_env(self, normalized: np.ndarray) -> dict[str, float]:
        a = np.asarray(normalized, dtype=np.float32).reshape(-1)
        if len(a) == 1:  # agent-level action fallback
            a = np.array([a[0], 0.0, 0.0], dtype=np.float32)
        if len(a) < 6:
            a = np.pad(a, (0, 6-len(a)), constant_values=0.0)
        return {
            "ess_power_kw": float(np.clip(a[0], -1.0, 1.0) * self.ess_max),
            "ev_power_kw": float(np.clip(a[1], -1.0, 1.0) * self.ev_max),
            "dr_kw": float((np.clip(a[2], -1.0, 1.0) + 1.0) * 0.5 * self.dr_max),
            "upload_priority": float((np.clip(a[3], -1.0, 1.0) + 1.0) * 0.5),
            "edge_cpu_fraction": float((np.clip(a[4], -1.0, 1.0) + 1.0) * 0.5),
            "offload_ratio": float((np.clip(a[5], -1.0, 1.0) + 1.0) * 0.35),
        }

    def to_env_from_agents(self, normalized: np.ndarray) -> dict[str, float]:
        # three scalar agent actions: ESS, EV, DR.
        a = np.asarray(normalized, dtype=np.float32).reshape(-1)
        if len(a) < 6:
            a = np.pad(a, (0, 6-len(a)), constant_values=0.0)
        return {
            "ess_power_kw": float(np.clip(a[0], -1.0, 1.0) * self.ess_max),
            "ev_power_kw": float(np.clip(a[1], -1.0, 1.0) * self.ev_max),
            "dr_kw": float((np.clip(a[2], -1.0, 1.0) + 1.0) * 0.5 * self.dr_max),
            "upload_priority": float((np.clip(a[3], -1.0, 1.0) + 1.0) * 0.5),
            "edge_cpu_fraction": float((np.clip(a[4], -1.0, 1.0) + 1.0) * 0.5),
            "offload_ratio": float((np.clip(a[5], -1.0, 1.0) + 1.0) * 0.35),
        }


class MLP(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, output_activation: str | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_activation = output_activation

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.net(x)
        if self.output_activation == "tanh":
            return torch.tanh(y)
        return y


class GaussianActor(nn.Module):  # type: ignore[misc]
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, init_std: float = 0.35) -> None:
        super().__init__()
        self.mean = MLP(obs_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(init_std))

    def distribution(self, obs: "torch.Tensor") -> Any:
        mean = torch.tanh(self.mean(obs))
        std = torch.exp(self.log_std).clamp(0.05, 1.0)
        return Normal(mean, std)

    def sample(self, obs: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        dist = self.distribution(obs)
        raw = dist.rsample()
        action = torch.clamp(raw, -1.0, 1.0)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def deterministic(self, obs: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.mean(obs))

    def log_prob_entropy(self, obs: "torch.Tensor", action: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class DeterministicActor(nn.Module):  # type: ignore[misc]
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_dim, output_activation="tanh")

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        return self.net(obs)


@dataclass
class TrajectoryBatch:
    global_states: list[np.ndarray] = field(default_factory=list)
    agent_obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.global_states.clear(); self.agent_obs.clear(); self.actions.clear()
        self.log_probs.clear(); self.rewards.clear(); self.dones.clear(); self.values.clear()


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 42) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.data: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]] = []

    def add(self, state: np.ndarray, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: float) -> None:
        item = (state.astype(np.float32), obs.astype(np.float32), action.astype(np.float32), float(reward), next_obs.astype(np.float32), float(done))
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.choice(len(self.data), size=min(batch_size, len(self.data)), replace=False)
        batch = [self.data[i] for i in idx]
        s, o, a, r, no, d = zip(*batch)
        return np.stack(s), np.stack(o), np.stack(a), np.asarray(r, dtype=np.float32), np.stack(no), np.asarray(d, dtype=np.float32)


def compute_gae(rewards: list[float], dones: list[float], values: list[float], gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_adv = 0.0
    values_ext = values + [0.0]
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + np.asarray(values, dtype=np.float32)
    adv_std = float(advantages.std())
    if adv_std > 1e-8:
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


class NeuralEMSBase:
    """Common evaluation logic for trained neural EMS policies."""

    name = "neural_ems"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for neural DT-C3 algorithms")
        self.config = config or RLTrainConfig()
        self.config.device = resolve_torch_device(self.config.device)
        set_global_seed(self.config.seed)
        self.device = torch.device(self.config.device)
        self.encoder = FeatureEncoder(self.config.communication_aware, self.config.use_aoi)
        self.mapper = ActionMapper()
        self.shield = shield or SafetyShield()
        self.trace = RLTrainingTrace(method=self.name, seed=self.config.seed)

    def train(self, env_factory: Callable[[], PowerIoTVPPEnvironment], episodes: int | None = None) -> RLTrainingTrace:
        raise NotImplementedError

    def normalized_action(self, state: dict[str, float], deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def act(self, state: dict[str, float]) -> dict[str, float]:
        action = self.mapper.to_env(self.normalized_action(state, deterministic=True))
        if self.config.safety_enabled:
            return self.shield.project(state, action).corrected_action
        return action

    def evaluate(self, env: PowerIoTVPPEnvironment) -> dict[str, Any]:
        state = env.reset()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        total_violations = 0
        total_curtailment = 0.0
        decision_times: list[float] = []
        grid_power: list[float] = []
        soc: list[float] = []
        pre_shield_violations = 0
        correction_magnitudes: list[float] = []
        while not done:
            t0 = time.perf_counter()
            raw_action = self.mapper.to_env(self.normalized_action(state, deterministic=True))
            action = raw_action
            if self.config.safety_enabled:
                report = self.shield.project(state, raw_action)
                action = report.corrected_action
                pre_shield_violations += report.violation_count
                correction_magnitudes.append(sum(abs(action.get(k, 0.0) - raw_action.get(k, 0.0)) for k in ACTION_KEYS))
            result = env.step(action, apply_safety=False)
            decision_times.append(time.perf_counter() - t0 + float(result.info.get("decision_time_s", 0.0)))
            state = result.state
            done = result.done
            total_reward += result.reward
            total_cost += float(result.info.get("cost", 0.0))
            total_violations += int(result.info.get("constraint_violations", 0))
            total_curtailment += float(result.info.get("curtailment_kw", 0.0)) * env.config.dt_hours
            grid_power.append(float(result.info.get("grid_power_kw", 0.0)))
            if state:
                soc.append(float(state.get("ess_soc", 0.5)))
        out = {
            "method": self.name,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "constraint_violations": total_violations,
            "curtailment_kwh": total_curtailment,
            "avg_aoi": env.channel.average_aoi(),
            "dt_reconstruction_error": env.twin.mean_reconstruction_error(),
            "avg_decision_time_s": float(np.mean(decision_times)) if decision_times else 0.0,
            "grid_power_std_kw": float(np.std(grid_power)) if grid_power else 0.0,
            "soc_min": float(np.min(soc)) if soc else 0.0,
            "soc_max": float(np.max(soc)) if soc else 0.0,
            "pre_shield_violations": pre_shield_violations,
            "safety_correction_kw_mean": float(np.mean(correction_magnitudes)) if correction_magnitudes else 0.0,
            "avg_delay_steps": env.channel.stats().get("avg_delay_steps", 0.0),
            "delivery_ratio": env.channel.stats().get("delivery_ratio", 1.0),
            "communication_overhead_kb": env.channel.stats().get("communication_overhead_kb", 0.0),
            "dt_variable_rmse": env.twin.variable_rmse(),
            "computation_latency_s": float(env.last_info.get("computation_latency_s", 0.0)) if env.last_info else 0.0,
            "c3_total_penalty": float(env.last_info.get("c3_total_penalty", 0.0)) if env.last_info else 0.0,
        }
        out.update(self.trace.summary())
        return out


class MAPPOEMS(NeuralEMSBase):
    """MAPPO-style CTDE baseline.

    Decentralized actors receive local/global observations with an agent ID.
    A centralized critic evaluates the full VPP state, implementing CTDE.
    """

    name = "mappo_ctde"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        super().__init__(config, shield)
        self.agent_obs_dim = self.encoder.dim + len(AGENT_NAMES)
        self.actors = nn.ModuleList([
            GaussianActor(self.agent_obs_dim, 1, self.config.hidden_dim, self.config.exploration_std) for _ in AGENT_NAMES
        ]).to(self.device)
        self.critic = MLP(self.encoder.dim, 1, self.config.hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=self.config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

    def _sample_action(self, state: dict[str, float]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        global_state = self.encoder.encode(state)
        agent_obs = np.stack([self.encoder.agent_obs(state, i) for i in range(len(AGENT_NAMES))])
        obs_t = torch.tensor(agent_obs, dtype=torch.float32, device=self.device)
        actions = []
        log_probs = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                a, lp, _ = actor.sample(obs_t[i : i + 1])
                actions.append(float(a.cpu().numpy().reshape(-1)[0]))
                log_probs.append(float(lp.cpu().numpy().reshape(-1)[0]))
            value = float(self.critic(torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)).cpu().numpy().reshape(-1)[0])
        return np.asarray(actions, dtype=np.float32), np.asarray(log_probs, dtype=np.float32), value, global_state, agent_obs

    def normalized_action(self, state: dict[str, float], deterministic: bool = True) -> np.ndarray:
        obs = np.stack([self.encoder.agent_obs(state, i) for i in range(len(AGENT_NAMES))])
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                if deterministic:
                    a = actor.deterministic(obs_t[i : i + 1])
                else:
                    a, _, _ = actor.sample(obs_t[i : i + 1])
                actions.append(float(a.cpu().numpy().reshape(-1)[0]))
        return np.asarray(actions, dtype=np.float32)

    def train(self, env_factory: Callable[[], PowerIoTVPPEnvironment], episodes: int | None = None) -> RLTrainingTrace:
        eps = episodes or self.config.episodes
        self.trace = RLTrainingTrace(method=self.name, seed=self.config.seed)
        for _ep in range(eps):
            traj = TrajectoryBatch()
            env = env_factory()
            state = env.reset()
            done = False
            ep_reward = 0.0
            ep_cost = 0.0
            ep_violations = 0
            while not done:
                norm_a, logp, value, gstate, aobs = self._sample_action(state)
                action = self.mapper.to_env_from_agents(norm_a)
                if self.config.safety_enabled:
                    action = self.shield.project(state, action).corrected_action
                result = env.step(action, apply_safety=False)
                ep_reward += result.reward
                ep_cost += float(result.info.get("cost", 0.0))
                ep_violations += int(result.info.get("constraint_violations", 0))
                traj.global_states.append(gstate)
                traj.agent_obs.append(aobs)
                traj.actions.append(norm_a)
                traj.log_probs.append(logp)
                traj.rewards.append(float(result.reward) / 100.0)  # scale rewards for stable updates
                traj.dones.append(float(result.done))
                traj.values.append(value)
                state = result.state
                done = result.done
            self._update(traj)
            self.trace.episode_rewards.append(ep_reward)
            self.trace.episode_costs.append(ep_cost)
            self.trace.episode_violations.append(ep_violations)
        return self.trace

    def _update(self, traj: TrajectoryBatch) -> None:
        if not traj.rewards:
            return
        t0 = time.perf_counter()
        adv, ret = compute_gae(traj.rewards, traj.dones, traj.values, self.config.gamma, self.config.gae_lambda)
        states = torch.tensor(np.stack(traj.global_states), dtype=torch.float32, device=self.device)
        agent_obs = torch.tensor(np.stack(traj.agent_obs), dtype=torch.float32, device=self.device)  # T,A,D
        actions = torch.tensor(np.stack(traj.actions), dtype=torch.float32, device=self.device)  # T,A
        old_logp = torch.tensor(np.stack(traj.log_probs), dtype=torch.float32, device=self.device)  # T,A
        advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret, dtype=torch.float32, device=self.device)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        for _ in range(self.config.update_epochs):
            logps = []
            ents = []
            for i, actor in enumerate(self.actors):
                lp, ent = actor.log_prob_entropy(agent_obs[:, i, :], actions[:, i : i + 1])
                logps.append(lp)
                ents.append(ent)
            logp = torch.stack(logps, dim=1)
            entropy = torch.stack(ents, dim=1).mean()
            ratio = torch.exp(logp - old_logp)
            adv_expand = advantages.unsqueeze(1).expand_as(ratio)
            surr1 = ratio * adv_expand
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv_expand
            actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy
            values = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(values, returns)
            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 1.0)
            self.actor_opt.step()
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
        self.trace.actor_losses.append(float(np.mean(actor_losses)))
        self.trace.critic_losses.append(float(np.mean(critic_losses)))
        self.trace.entropy.append(float(np.mean(entropies)))
        self.trace.update_times_s.append(time.perf_counter() - t0)


class PPOEMS(MAPPOEMS):
    """Single-agent PPO baseline using the same Actor-Critic machinery."""

    name = "ppo_actor_critic"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        NeuralEMSBase.__init__(self, config, shield)
        self.actor = GaussianActor(self.encoder.dim, self.mapper.dim, self.config.hidden_dim, self.config.exploration_std).to(self.device)
        self.critic = MLP(self.encoder.dim, 1, self.config.hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

    def normalized_action(self, state: dict[str, float], deterministic: bool = True) -> np.ndarray:
        obs = torch.tensor(self.encoder.encode(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(obs)
            else:
                a, _, _ = self.actor.sample(obs)
        return a.cpu().numpy().reshape(-1).astype(np.float32)

    def train(self, env_factory: Callable[[], PowerIoTVPPEnvironment], episodes: int | None = None) -> RLTrainingTrace:
        eps = episodes or self.config.episodes
        self.trace = RLTrainingTrace(method=self.name, seed=self.config.seed)
        for _ep in range(eps):
            states=[]; actions=[]; logps=[]; rewards=[]; dones=[]; values=[]
            env=env_factory(); state=env.reset(); done=False
            ep_reward=ep_cost=0.0; ep_viol=0
            while not done:
                encoded=self.encoder.encode(state)
                obs_t=torch.tensor(encoded,dtype=torch.float32,device=self.device).unsqueeze(0)
                with torch.no_grad():
                    a,lp,_=self.actor.sample(obs_t)
                    val=float(self.critic(obs_t).cpu().numpy().reshape(-1)[0])
                norm=a.cpu().numpy().reshape(-1).astype(np.float32)
                env_action=self.mapper.to_env(norm)
                if self.config.safety_enabled:
                    env_action=self.shield.project(state, env_action).corrected_action
                result=env.step(env_action, apply_safety=False)
                states.append(encoded); actions.append(norm); logps.append(float(lp.cpu().numpy().reshape(-1)[0])); rewards.append(result.reward/100.0); dones.append(float(result.done)); values.append(val)
                ep_reward+=result.reward; ep_cost+=float(result.info.get('cost',0)); ep_viol+=int(result.info.get('constraint_violations',0))
                state=result.state; done=result.done
            self._update_single(states,actions,logps,rewards,dones,values)
            self.trace.episode_rewards.append(ep_reward); self.trace.episode_costs.append(ep_cost); self.trace.episode_violations.append(ep_viol)
        return self.trace

    def _update_single(self, states, actions, old_logps, rewards, dones, values) -> None:
        t0=time.perf_counter()
        adv,ret=compute_gae(rewards,dones,values,self.config.gamma,self.config.gae_lambda)
        s=torch.tensor(np.stack(states),dtype=torch.float32,device=self.device)
        a=torch.tensor(np.stack(actions),dtype=torch.float32,device=self.device)
        old=torch.tensor(np.asarray(old_logps,dtype=np.float32),dtype=torch.float32,device=self.device)
        adv_t=torch.tensor(adv,dtype=torch.float32,device=self.device)
        ret_t=torch.tensor(ret,dtype=torch.float32,device=self.device)
        als=[]; cls=[]; ents=[]
        for _ in range(self.config.update_epochs):
            lp,ent=self.actor.log_prob_entropy(s,a)
            ratio=torch.exp(lp-old)
            surr1=ratio*adv_t; surr2=torch.clamp(ratio,1-self.config.clip_ratio,1+self.config.clip_ratio)*adv_t
            actor_loss=-torch.min(surr1,surr2).mean()-self.config.entropy_coef*ent.mean()
            critic_loss=F.mse_loss(self.critic(s).squeeze(-1),ret_t)
            self.actor_opt.zero_grad(); actor_loss.backward(retain_graph=True); torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.actor_opt.step()
            self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.critic_opt.step()
            als.append(float(actor_loss.detach().cpu())); cls.append(float(critic_loss.detach().cpu())); ents.append(float(ent.mean().detach().cpu()))
        self.trace.actor_losses.append(float(np.mean(als))); self.trace.critic_losses.append(float(np.mean(cls))); self.trace.entropy.append(float(np.mean(ents))); self.trace.update_times_s.append(time.perf_counter()-t0)


class IACEMS(MAPPOEMS):
    """Independent actor-critic baseline.

    It uses decentralized actors and decentralized value functions.  It is a
    deliberately weaker baseline than CTDE MAPPO.
    """

    name = "iac_actor_critic"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        NeuralEMSBase.__init__(self, config, shield)
        self.agent_obs_dim = self.encoder.dim + len(AGENT_NAMES)
        self.actors = nn.ModuleList([GaussianActor(self.agent_obs_dim, 1, self.config.hidden_dim, self.config.exploration_std) for _ in AGENT_NAMES]).to(self.device)
        self.critics = nn.ModuleList([MLP(self.agent_obs_dim, 1, self.config.hidden_dim) for _ in AGENT_NAMES]).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=self.config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=self.config.lr_critic)

    def _sample_action(self, state: dict[str, float]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        global_state = self.encoder.encode(state)
        agent_obs = np.stack([self.encoder.agent_obs(state, i) for i in range(len(AGENT_NAMES))])
        obs_t = torch.tensor(agent_obs, dtype=torch.float32, device=self.device)
        actions=[]; log_probs=[]; vals=[]
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                a,lp,_=actor.sample(obs_t[i:i+1]); actions.append(float(a.cpu().numpy().reshape(-1)[0])); log_probs.append(float(lp.cpu().numpy().reshape(-1)[0])); vals.append(float(self.critics[i](obs_t[i:i+1]).cpu().numpy().reshape(-1)[0]))
        return np.asarray(actions,dtype=np.float32), np.asarray(log_probs,dtype=np.float32), float(np.mean(vals)), global_state, agent_obs

    def _update(self, traj: TrajectoryBatch) -> None:
        if not traj.rewards:
            return
        t0=time.perf_counter(); adv,ret=compute_gae(traj.rewards,traj.dones,traj.values,self.config.gamma,self.config.gae_lambda)
        agent_obs=torch.tensor(np.stack(traj.agent_obs),dtype=torch.float32,device=self.device); actions=torch.tensor(np.stack(traj.actions),dtype=torch.float32,device=self.device); old_logp=torch.tensor(np.stack(traj.log_probs),dtype=torch.float32,device=self.device)
        advantages=torch.tensor(adv,dtype=torch.float32,device=self.device); returns=torch.tensor(ret,dtype=torch.float32,device=self.device)
        als=[]; cls=[]; ents=[]
        for _ in range(self.config.update_epochs):
            actor_losses=[]; critic_losses=[]; entropies=[]
            for i,actor in enumerate(self.actors):
                lp,ent=actor.log_prob_entropy(agent_obs[:,i,:],actions[:,i:i+1]); ratio=torch.exp(lp-old_logp[:,i]); surr1=ratio*advantages; surr2=torch.clamp(ratio,1-self.config.clip_ratio,1+self.config.clip_ratio)*advantages
                actor_losses.append(-torch.min(surr1,surr2).mean()-self.config.entropy_coef*ent.mean())
                critic_losses.append(F.mse_loss(self.critics[i](agent_obs[:,i,:]).squeeze(-1),returns)); entropies.append(ent.mean())
            actor_loss=torch.stack(actor_losses).mean(); critic_loss=torch.stack(critic_losses).mean(); entropy=torch.stack(entropies).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(retain_graph=True); torch.nn.utils.clip_grad_norm_(self.actors.parameters(),1.0); self.actor_opt.step()
            self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critics.parameters(),1.0); self.critic_opt.step()
            als.append(float(actor_loss.detach().cpu())); cls.append(float(critic_loss.detach().cpu())); ents.append(float(entropy.detach().cpu()))
        self.trace.actor_losses.append(float(np.mean(als))); self.trace.critic_losses.append(float(np.mean(cls))); self.trace.entropy.append(float(np.mean(ents))); self.trace.update_times_s.append(time.perf_counter()-t0)


class MADDPGEMS(NeuralEMSBase):
    """MADDPG-style off-policy baseline with centralized critics."""

    name = "maddpg_ctde"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        super().__init__(config, shield)
        self.agent_obs_dim = self.encoder.dim + len(AGENT_NAMES)
        self.actors = nn.ModuleList([DeterministicActor(self.agent_obs_dim, 1, self.config.hidden_dim) for _ in AGENT_NAMES]).to(self.device)
        self.target_actors = nn.ModuleList([DeterministicActor(self.agent_obs_dim, 1, self.config.hidden_dim) for _ in AGENT_NAMES]).to(self.device)
        self.critic = MLP(self.encoder.dim + len(AGENT_NAMES), 1, self.config.hidden_dim).to(self.device)
        self.target_critic = MLP(self.encoder.dim + len(AGENT_NAMES), 1, self.config.hidden_dim).to(self.device)
        self.target_actors.load_state_dict(self.actors.state_dict()); self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt=torch.optim.Adam(self.actors.parameters(),lr=self.config.lr_actor); self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=self.config.lr_critic)
        self.buffer=ReplayBuffer(seed=self.config.seed)
        self.tau=0.01

    def normalized_action(self, state: dict[str, float], deterministic: bool = True) -> np.ndarray:
        obs=np.stack([self.encoder.agent_obs(state,i) for i in range(len(AGENT_NAMES))]); obs_t=torch.tensor(obs,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            acts=[]
            for i,a in enumerate(self.actors):
                val=float(a(obs_t[i:i+1]).cpu().numpy().reshape(-1)[0])
                if not deterministic:
                    val += float(np.random.normal(0,self.config.exploration_std))
                acts.append(np.clip(val,-1,1))
        return np.asarray(acts,dtype=np.float32)

    def train(self, env_factory: Callable[[], PowerIoTVPPEnvironment], episodes: int | None = None) -> RLTrainingTrace:
        eps=episodes or self.config.episodes; self.trace=RLTrainingTrace(method=self.name,seed=self.config.seed)
        for _ep in range(eps):
            env=env_factory(); state=env.reset(); done=False; ep_reward=ep_cost=0.0; ep_viol=0; als=[]; cls=[]
            while not done:
                g=self.encoder.encode(state); obs=np.stack([self.encoder.agent_obs(state,i) for i in range(len(AGENT_NAMES))])
                norm=self.normalized_action(state,deterministic=False); action=self.mapper.to_env_from_agents(norm)
                if self.config.safety_enabled: action=self.shield.project(state,action).corrected_action
                result=env.step(action,apply_safety=False); next_obs=np.stack([self.encoder.agent_obs(result.state,i) if result.state else np.zeros(self.agent_obs_dim,dtype=np.float32) for i in range(len(AGENT_NAMES))])
                self.buffer.add(g,obs,norm,float(result.reward)/100.0,next_obs,float(result.done)); state=result.state; done=result.done; ep_reward+=result.reward; ep_cost+=float(result.info.get('cost',0)); ep_viol+=int(result.info.get('constraint_violations',0))
                if len(self.buffer)>=max(8,min(self.config.batch_size,32)):
                    al,cl=self._update(); als.append(al); cls.append(cl)
            self.trace.episode_rewards.append(ep_reward); self.trace.episode_costs.append(ep_cost); self.trace.episode_violations.append(ep_viol); self.trace.actor_losses.append(float(np.mean(als)) if als else 0.0); self.trace.critic_losses.append(float(np.mean(cls)) if cls else 0.0); self.trace.entropy.append(0.0)
        return self.trace

    def _update(self) -> tuple[float,float]:
        t0=time.perf_counter(); states,obs,acts,rews,next_obs,dones=self.buffer.sample(self.config.batch_size)
        s=torch.tensor(states,dtype=torch.float32,device=self.device); o=torch.tensor(obs,dtype=torch.float32,device=self.device); a=torch.tensor(acts,dtype=torch.float32,device=self.device); r=torch.tensor(rews,dtype=torch.float32,device=self.device); no=torch.tensor(next_obs,dtype=torch.float32,device=self.device); d=torch.tensor(dones,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            next_actions=[]
            for i,ta in enumerate(self.target_actors): next_actions.append(ta(no[:,i,:]))
            na=torch.cat(next_actions,dim=1); q_next=self.target_critic(torch.cat([s,na],dim=1)).squeeze(-1); target=r+self.config.gamma*(1-d)*q_next
        q=self.critic(torch.cat([s,a],dim=1)).squeeze(-1); critic_loss=F.mse_loss(q,target)
        self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.critic_opt.step()
        pred_actions=[]
        for i,actor in enumerate(self.actors): pred_actions.append(actor(o[:,i,:]))
        pa=torch.cat(pred_actions,dim=1); actor_loss=-self.critic(torch.cat([s,pa],dim=1)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actors.parameters(),1.0); self.actor_opt.step()
        self._soft_update(); self.trace.update_times_s.append(time.perf_counter()-t0)
        return float(actor_loss.detach().cpu()),float(critic_loss.detach().cpu())

    def _soft_update(self) -> None:
        for target,src in zip(self.target_actors.parameters(),self.actors.parameters()): target.data.mul_(1-self.tau).add_(self.tau*src.data)
        for target,src in zip(self.target_critic.parameters(),self.critic.parameters()): target.data.mul_(1-self.tau).add_(self.tau*src.data)


class NoAoIMAPPOEMS(MAPPOEMS):
    """Ablation: communication-aware MAPPO but AoI feature removed."""

    name = "without_aoi"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        cfg = config or RLTrainConfig()
        cfg.use_aoi = False
        super().__init__(cfg, shield)


class DTC3MAPPOEMS(MAPPOEMS):
    """Proposed DT-C3 Safe-MARL implementation.

    It inherits MAPPO CTDE and keeps DT/communication/safety enabled.  This is
    the algorithm to report as DT-C3-SafeMARL after sufficient training.
    """

    name = "dt_c3_safemarl_mappo"


def run_multi_seed_training(
    method_factory: Callable[[int], NeuralEMSBase],
    env_factory: Callable[[int], PowerIoTVPPEnvironment],
    seeds: Iterable[int],
    episodes: int,
) -> tuple[list[dict[str, Any]], list[RLTrainingTrace]]:
    rows: list[dict[str, Any]] = []
    traces: list[RLTrainingTrace] = []
    for seed in seeds:
        method = method_factory(int(seed))
        trace = method.train(lambda s=int(seed): env_factory(s), episodes=episodes)
        metrics = method.evaluate(env_factory(int(seed) + 10_000))
        metrics["seed"] = int(seed)
        rows.append(metrics)
        traces.append(trace)
    return rows, traces

# ---------------------------------------------------------------------------
# Final paper-minimum algorithm upgrades
# ---------------------------------------------------------------------------
from pathlib import Path as _Path


class MAPPOEMS(MAPPOEMS):  # type: ignore[no-redef]
    """Paper-minimum MAPPO with mini-batches, value clipping and checkpoints.

    This class intentionally shadows the compact prototype above while keeping
    the public class name unchanged for all experiment scripts.
    """

    name = "mappo_ctde_paper"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        super().__init__(config, shield)
        self.checkpoint_dir = _Path("output/dt_c3_checkpoints")
        self.best_reward = -1e18

    def train(self, env_factory: Callable[[], PowerIoTVPPEnvironment], episodes: int | None = None) -> RLTrainingTrace:
        eps = episodes or self.config.episodes
        self.trace = RLTrainingTrace(method=self.name, seed=self.config.seed)
        for ep in range(eps):
            traj = TrajectoryBatch()
            env = env_factory(); state = env.reset(); done = False
            ep_reward = 0.0; ep_cost = 0.0; ep_violations = 0
            while not done:
                norm_a, logp, value, gstate, aobs = self._sample_action(state)
                action = self.mapper.to_env_from_agents(norm_a)
                if self.config.safety_enabled:
                    action = self.shield.project(state, action).corrected_action
                # DT preview risk is already included in env reward.  This keeps
                # policy learning coupled to decision-level digital-twin safety.
                result = env.step(action, apply_safety=False)
                ep_reward += result.reward
                ep_cost += float(result.info.get("cost", 0.0))
                ep_violations += int(result.info.get("constraint_violations", 0))
                traj.global_states.append(gstate)
                traj.agent_obs.append(aobs)
                traj.actions.append(norm_a)
                traj.log_probs.append(logp)
                traj.rewards.append(float(result.reward) / 100.0)
                traj.dones.append(float(result.done))
                traj.values.append(value)
                state = result.state; done = result.done
            self._update(traj, episode_index=ep)
            self.trace.episode_rewards.append(ep_reward)
            self.trace.episode_costs.append(ep_cost)
            self.trace.episode_violations.append(ep_violations)
            if self.config.save_checkpoints and ep_reward > self.best_reward:
                self.best_reward = ep_reward
                self.save_checkpoint(self.checkpoint_dir / f"{self.name}_seed{self.config.seed}_best.pt")
        return self.trace

    def _update(self, traj: TrajectoryBatch, episode_index: int = 0) -> None:
        if not traj.rewards:
            return
        t0 = time.perf_counter()
        adv, ret = compute_gae(traj.rewards, traj.dones, traj.values, self.config.gamma, self.config.gae_lambda)
        states = torch.tensor(np.stack(traj.global_states), dtype=torch.float32, device=self.device)
        agent_obs = torch.tensor(np.stack(traj.agent_obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(traj.actions), dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(np.stack(traj.log_probs), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            old_values = self.critic(states).squeeze(-1)
        T = states.shape[0]
        batch_size = max(1, min(int(self.config.batch_size), T))
        idx_all = np.arange(T)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        entropy_coef = float(self.config.entropy_coef) * max(0.15, 1.0 - episode_index / max(1, self.config.episodes))
        for _ in range(self.config.update_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, T, batch_size):
                idx_np = idx_all[start : start + batch_size]
                idx = torch.tensor(idx_np, dtype=torch.long, device=self.device)
                b_obs = agent_obs[idx]; b_actions = actions[idx]; b_old_logp = old_logp[idx]
                b_adv = advantages[idx]; b_returns = returns[idx]; b_states = states[idx]; b_old_values = old_values[idx]
                logps=[]; ents=[]
                for i, actor in enumerate(self.actors):
                    lp, ent = actor.log_prob_entropy(b_obs[:, i, :], b_actions[:, i : i + 1])
                    logps.append(lp); ents.append(ent)
                logp = torch.stack(logps, dim=1)
                entropy = torch.stack(ents, dim=1).mean()
                ratio = torch.exp(logp - b_old_logp)
                adv_expand = b_adv.unsqueeze(1).expand_as(ratio)
                surr1 = ratio * adv_expand
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv_expand
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                values = self.critic(b_states).squeeze(-1)
                value_clipped = b_old_values + torch.clamp(values - b_old_values, -self.config.clip_ratio, self.config.clip_ratio)
                critic_loss_unclipped = (values - b_returns).pow(2)
                critic_loss_clipped = (value_clipped - b_returns).pow(2)
                critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                self.actor_opt.zero_grad(); actor_loss.backward(retain_graph=True); torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 1.0); self.actor_opt.step()
                self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()
                actor_losses.append(float(actor_loss.detach().cpu())); critic_losses.append(float(critic_loss.detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        self.trace.actor_losses.append(float(np.mean(actor_losses)) if actor_losses else 0.0)
        self.trace.critic_losses.append(float(np.mean(critic_losses)) if critic_losses else 0.0)
        self.trace.entropy.append(float(np.mean(entropies)) if entropies else 0.0)
        self.trace.update_times_s.append(time.perf_counter() - t0)

    def save_checkpoint(self, path: _Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "method": self.name,
            "seed": self.config.seed,
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "trace": self.trace.__dict__,
        }, path)

    def load_checkpoint(self, path: str | _Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])


class PPOEMS(MAPPOEMS):  # type: ignore[no-redef]
    name = "ppo_actor_critic_paper"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        super().__init__(config, shield)
        self.actors = nn.ModuleList([GaussianActor(self.encoder.dim, self.mapper.dim, self.config.hidden_dim, self.config.exploration_std)]).to(self.device)
        self.critic = MLP(self.encoder.dim, 1, self.config.hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=self.config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

    def _sample_action(self, state: dict[str, float]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        global_state = self.encoder.encode(state)
        obs_t = torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, lp, _ = self.actors[0].sample(obs_t)
            value = float(self.critic(obs_t).cpu().numpy().reshape(-1)[0])
        action = a.cpu().numpy().reshape(-1).astype(np.float32)
        logp = np.repeat(float(lp.cpu().numpy().reshape(-1)[0]) / self.mapper.dim, self.mapper.dim).astype(np.float32)
        agent_obs = np.stack([self.encoder.agent_obs(state, i) for i in range(len(AGENT_NAMES))])
        return action, logp, value, global_state, agent_obs

    def normalized_action(self, state: dict[str, float], deterministic: bool = True) -> np.ndarray:
        obs = torch.tensor(self.encoder.encode(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actors[0].deterministic(obs)
            else:
                a, _, _ = self.actors[0].sample(obs)
        return a.cpu().numpy().reshape(-1).astype(np.float32)

    def _update(self, traj: TrajectoryBatch, episode_index: int = 0) -> None:
        # Reuse PPO single actor by temporarily reshaping stored actions/logps.
        if not traj.rewards:
            return
        t0=time.perf_counter(); adv,ret=compute_gae(traj.rewards,traj.dones,traj.values,self.config.gamma,self.config.gae_lambda)
        states=torch.tensor(np.stack(traj.global_states),dtype=torch.float32,device=self.device)
        actions=torch.tensor(np.stack(traj.actions),dtype=torch.float32,device=self.device)
        old_logp=torch.tensor(np.stack(traj.log_probs).sum(axis=1),dtype=torch.float32,device=self.device)
        advantages=torch.tensor(adv,dtype=torch.float32,device=self.device); returns=torch.tensor(ret,dtype=torch.float32,device=self.device)
        with torch.no_grad(): old_values=self.critic(states).squeeze(-1)
        T=states.shape[0]; bs=max(1,min(self.config.batch_size,T)); idx_all=np.arange(T); als=[]; cls=[]; ents=[]
        entropy_coef=float(self.config.entropy_coef)*max(0.15,1-episode_index/max(1,self.config.episodes))
        for _ in range(self.config.update_epochs):
            np.random.shuffle(idx_all)
            for start in range(0,T,bs):
                idx=torch.tensor(idx_all[start:start+bs],dtype=torch.long,device=self.device)
                lp,ent=self.actors[0].log_prob_entropy(states[idx],actions[idx])
                ratio=torch.exp(lp-old_logp[idx]); s1=ratio*advantages[idx]; s2=torch.clamp(ratio,1-self.config.clip_ratio,1+self.config.clip_ratio)*advantages[idx]
                actor_loss=-torch.min(s1,s2).mean()-entropy_coef*ent.mean()
                values=self.critic(states[idx]).squeeze(-1); vclip=old_values[idx]+torch.clamp(values-old_values[idx],-self.config.clip_ratio,self.config.clip_ratio)
                critic_loss=0.5*torch.max((values-returns[idx]).pow(2),(vclip-returns[idx]).pow(2)).mean()
                self.actor_opt.zero_grad(); actor_loss.backward(retain_graph=True); torch.nn.utils.clip_grad_norm_(self.actors.parameters(),1.0); self.actor_opt.step()
                self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.critic_opt.step()
                als.append(float(actor_loss.detach().cpu())); cls.append(float(critic_loss.detach().cpu())); ents.append(float(ent.mean().detach().cpu()))
        self.trace.actor_losses.append(float(np.mean(als)) if als else 0.0); self.trace.critic_losses.append(float(np.mean(cls)) if cls else 0.0); self.trace.entropy.append(float(np.mean(ents)) if ents else 0.0); self.trace.update_times_s.append(time.perf_counter()-t0)


class IACEMS(MAPPOEMS):  # type: ignore[no-redef]
    name = "iac_independent_actor_critic_paper"


class DTC3MAPPOEMS(MAPPOEMS):  # type: ignore[no-redef]
    name = "dt_c3_safemarl_mappo_paper"


class NoAoIMAPPOEMS(MAPPOEMS):  # type: ignore[no-redef]
    name = "without_aoi"
    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        cfg = config or RLTrainConfig(); cfg.use_aoi = False
        super().__init__(cfg, shield)


class MADDPGEMS(MADDPGEMS):  # type: ignore[no-redef]
    """MADDPG with exploration decay, delayed actor updates and checkpoints."""

    name = "maddpg_ctde_paper"

    def __init__(self, config: RLTrainConfig | None = None, shield: SafetyShield | None = None) -> None:
        super().__init__(config, shield)
        self.tau = 0.005
        self.policy_delay = 2
        self.update_step = 0
        self.noise_initial = self.config.exploration_std
        self.noise_final = 0.05
        self.checkpoint_dir = _Path("output/dt_c3_checkpoints")
        self.best_reward = -1e18

    def normalized_action(self, state: dict[str, float], deterministic: bool = True) -> np.ndarray:
        obs=np.stack([self.encoder.agent_obs(state,i) for i in range(len(AGENT_NAMES))]); obs_t=torch.tensor(obs,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            acts=[]
            for i,a in enumerate(self.actors):
                val=float(a(obs_t[i:i+1]).cpu().numpy().reshape(-1)[0])
                if not deterministic:
                    progress=min(1.0,self.update_step/max(1,self.config.episodes*10))
                    sigma=self.noise_initial*(1-progress)+self.noise_final*progress
                    val += float(np.random.normal(0,sigma))
                acts.append(np.clip(val,-1,1))
        return np.asarray(acts,dtype=np.float32)

    def _update(self) -> tuple[float,float]:
        self.update_step += 1
        states,obs,acts,rews,next_obs,dones=self.buffer.sample(self.config.batch_size)
        s=torch.tensor(states,dtype=torch.float32,device=self.device); o=torch.tensor(obs,dtype=torch.float32,device=self.device); a=torch.tensor(acts,dtype=torch.float32,device=self.device); r=torch.tensor(rews,dtype=torch.float32,device=self.device); no=torch.tensor(next_obs,dtype=torch.float32,device=self.device); d=torch.tensor(dones,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            na=torch.cat([ta(no[:,i,:]) for i,ta in enumerate(self.target_actors)],dim=1)
            q_next=self.target_critic(torch.cat([s,na],dim=1)).squeeze(-1)
            target=r+self.config.gamma*(1-d)*q_next
        q=self.critic(torch.cat([s,a],dim=1)).squeeze(-1); critic_loss=F.mse_loss(q,target)
        self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.critic_opt.step()
        actor_loss=torch.tensor(0.0,device=self.device)
        if self.update_step % self.policy_delay == 0:
            pa=torch.cat([actor(o[:,i,:]) for i,actor in enumerate(self.actors)],dim=1)
            actor_loss=-self.critic(torch.cat([s,pa],dim=1)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actors.parameters(),1.0); self.actor_opt.step()
            self._soft_update()
        return float(actor_loss.detach().cpu()),float(critic_loss.detach().cpu())

    def save_checkpoint(self, path: _Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actors":self.actors.state_dict(),"critic":self.critic.state_dict(),"target_actors":self.target_actors.state_dict(),"target_critic":self.target_critic.state_dict(),"trace":self.trace.__dict__},path)
