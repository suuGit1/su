"""统一配置；未知配置字段直接报错，避免参数拼写错误被忽略。"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import math


@dataclass
class Config:
    algorithm: str = 'mappo'
    episodes: int = 100
    horizon: int = 96
    dt_hours: float = 0.25
    seed: int = 1
    hidden_size: int = 64
    ppo_epoch: int = 5
    num_mini_batch: int = 1
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    reward_scale: float = 100.0
    violation_penalty: float = 100.0
    terminal_soc_penalty: float = 1000.0
    safety: bool = True
    digital_twin: bool = False
    delay_steps: int = 0
    packet_loss: float = 0.0
    train_csv: str | None = None
    device: str = 'cpu'
    threads: int = 1
    network_model: str = 'aggregate'
    dispatch_spec: str | None = None
    solver_time_limit: float = 30.0
    metrics_enabled: bool = False
    objective_scales: tuple = (100.0, 100.0, 100.0)
    objective_weights: tuple = (1.0, 0.0, 0.0)
    reserve_hours: float = 0.25
    synthetic_carbon_g_per_kwh: float | None = None
    resource_model: str = 'legacy'
    flex_spec: str | None = None
    ev_sessions_path: str | None = None

    def validate(self):
        if self.resource_model not in ('legacy', 'sessions_v1'):
            raise ValueError('resource_model 必须为 legacy 或 sessions_v1')
        if self.resource_model == 'sessions_v1' and self.network_model != 'ieee33':
            raise ValueError('会话资源模型要求 IEEE33')
        if self.resource_model == 'sessions_v1' and self.metrics_enabled and not math.isclose(self.reserve_hours,self.dt_hours):
            raise ValueError('会话模型当前备用持续时间必须等于一个调度步长')
        if self.network_model not in ('aggregate', 'ieee33'):
            raise ValueError('network_model 必须为 aggregate 或 ieee33')
        if not math.isfinite(self.solver_time_limit) or self.solver_time_limit <= 0:
            raise ValueError('求解时间必须为有限正数')
        if self.algorithm not in ('mappo', 'ippo', 'weighted_mappo'):
            raise ValueError('支持 mappo、ippo、weighted_mappo；Pareto 条件策略尚未接入，不能用别名替代')
        if type(self.metrics_enabled) is not bool:
            raise ValueError('metrics_enabled 必须为布尔值')
        if self.metrics_enabled and self.network_model != 'ieee33':
            raise ValueError('三目标评价要求 IEEE33 统一模型')
        if self.algorithm == 'weighted_mappo' and not self.metrics_enabled:
            raise ValueError('固定权重多目标方法必须启用三目标评价')
        for name in ('objective_scales', 'objective_weights'):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError(name + ' 必须为三个有限数')
        if min(self.objective_scales) <= 0 or min(self.objective_weights) < 0 or not math.isclose(sum(self.objective_weights), 1.0, abs_tol=1e-8):
            raise ValueError('目标尺度必须为正，偏好权重非负且和为 1')
        if not math.isfinite(self.reserve_hours) or self.reserve_hours <= 0:
            raise ValueError('备用持续时间必须为有限正数')
        if self.synthetic_carbon_g_per_kwh is not None and (not math.isfinite(self.synthetic_carbon_g_per_kwh) or self.synthetic_carbon_g_per_kwh < 0):
            raise ValueError('示例碳因子必须为有限非负数')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed 必须为非负整数')
        for key in ('safety', 'digital_twin'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key + ' 必须为布尔值')
        for key in ('episodes', 'horizon', 'hidden_size', 'ppo_epoch', 'num_mini_batch', 'threads'):
            v = getattr(self, key)
            if type(v) is not int or v < 1:
                raise ValueError(key + ' 必须为正整数')
        if self.horizon < 2 or self.num_mini_batch > 3 * self.horizon:
            raise ValueError('回合长度至少为 2，mini-batch 数不得超过样本数')
        if type(self.delay_steps) is not int or self.delay_steps < 0:
            raise ValueError('时延步数必须为非负整数')
        for key in ('lr', 'dt_hours', 'reward_scale', 'clip_param'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + ' 必须为有限正数')
        for key in ('gamma', 'gae_lambda', 'packet_loss'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(key + ' 必须在 0 到 1 之间')
        for key in ('violation_penalty', 'terminal_soc_penalty', 'entropy_coef'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(key + ' 必须为有限非负数')
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))).validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')
