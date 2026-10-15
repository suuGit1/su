"""可选协调器修订版；保留原协调器以便冻结实验复现。"""
import numpy as np
from .coordinator import TaskRiskCoordinator

# 对应 DT 的 SOC、EV需求、移峰积压、移峰累计、削减累计、负荷、电价、光伏、风电。
CHANNEL_TARGETS = ((0,), (1,), (2, 3, 5, 6), (4,), (7,), (8,))


class MaskedRiskCoordinator(TaskRiskCoordinator):
    def __init__(self, config, dispatch, flex, active_channels, mapped=True):
        super().__init__(config, dispatch, flex)
        mask = np.asarray(active_channels)
        if mask.shape != (6,) or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError('有效遥测掩码须为至少包含一个有效通道的六维布尔量')
        if not mask[2]:
            raise ValueError('通道2承载负荷与电价，不得因停用DR而屏蔽')
        self.active = mask.astype(bool)
        self.mapped = mapped

    def assess(self, observation):
        record = super().assess(observation)
        scores = np.asarray(record['risk_scores'])
        if self.mapped and self.config.risk_adaptive:
            widths = np.asarray(getattr(self, 'interval_width', np.zeros(9)), dtype=float)
            if widths.shape != (9,) or not np.isfinite(widths).all() or np.any(widths < 0):
                raise ValueError('DT区间宽度须为九个有限非负数')
            # 使用同一归一化尺度和上限；多目标通道取最大值，避免目标数量造成重复计权。
            mapped = np.minimum(5., np.array([widths[list(ids)].max() for ids in CHANNEL_TARGETS]) / .05)
            scores += mapped - np.asarray(record['interval_risk'])
            record['interval_risk'] = mapped.tolist()
        scores = np.where(self.active, scores, 0.)
        priority = np.where(self.active, 1. + scores, 0.)
        fraction = self.config.coordinator_reserved_fraction
        if self.config.risk_adaptive:
            fraction = min(.95, fraction + (1-fraction)*float(scores.max())/(1+float(scores.max())))
        record.update(contract='observation-task-risk-v2', revision='mapped' if self.mapped else 'mask_only',
                      active_channels=self.active.tolist(), risk_scores=scores.tolist(),
                      priority=priority.tolist(), reserved_fraction=float(fraction),
                      risk_channels=int(np.count_nonzero(scores)),
                      active_mean_aoi_steps=float(np.asarray(observation)[18:24][self.active].mean()))
        record['tasks'] = [t for t in record['tasks'] if t['kind'] != 'refresh_dt' or self.active[['ess','ev','dr_shift','dr_shed','pv','wind'].index(t['channel'])]]
        return record

    def allocate(self, observation, bw, cpu):
        record = self.assess(observation)
        if self.config.coordinator_mode != 'schedule':
            return np.asarray(bw), np.asarray(cpu), record
        priority = np.asarray(record['priority']); priority /= priority.sum()
        fraction = record['reserved_fraction']
        def blend(values):
            values = np.asarray(values, dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('资源权重须为六个有限非负数')
            values = np.where(self.active, values, 0.)
            base = values / values.sum() if values.sum() > 0 else self.active / self.active.sum()
            return (1-fraction)*base + fraction*priority
        return blend(bw), blend(cpu), record
