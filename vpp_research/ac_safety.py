"""现场 AC 校核与有限重求解；找不到可验证动作时终止仿真，不伪造后备保证。"""
from dataclasses import replace
import copy
import time
import numpy as np
from vpp_mappo.flex_environment import FlexVPPAdapter
from vpp_mappo.optimization import DispatchInfeasible


class ACCheckedFlex(FlexVPPAdapter):
    def __init__(self, config, csv_path=None, emergency=False):
        super().__init__(config, csv_path)
        self.emergency_enabled=bool(emergency)

    def reset(self,scenario_seed,profile_index=0):
        self.ac_rejected=False
        return super().reset(scenario_seed,profile_index)

    def step_physical(self,control):
        if self.ac_rejected:raise DispatchInfeasible('AC 已拒绝本回合，必须 reset 后再运行')
        if self.t>=self.config.horizon:raise RuntimeError('回合已结束')
        requested=np.asarray(control['action'] if isinstance(control,dict) else control,dtype=float)
        if requested.shape!=(6,) or not np.isfinite(requested).all():raise ValueError('AC 保护候选动作非法')
        row=self.row();pre=self.check(row,requested,self._allocate(requested[1]));start=time.perf_counter()
        original_spec=self.network.spec;original_smax=self.network.smax
        chosen=None;attempts=0;emergency_meta={}
        # 已通过约束的显式计划保持分车分配及 PCC 跟踪目标，执行前再次验收。
        if isinstance(control,dict):
            allocation=control.get('ev_kw',{})
            if set(allocation)<=set(self.remaining) and all(np.isfinite(v) for v in allocation.values()) and self.check(row,requested,allocation)==0:
                audit=self.network.audit(row,requested)
                if audit['ac_converged'] and audit['ac_violations']==0:chosen=control
        # 仅改变优化器的裕度，AC 验收始终使用原始物理边界。
        for margin in (() if chosen is not None else (0.,.002,.004,.006)):
            attempts+=1
            try:
                self.network.spec=replace(original_spec,voltage_min=original_spec.voltage_min+margin,voltage_max=original_spec.voltage_max-margin)
                self.network.smax=original_smax*(1-margin*5)
                plans,_=self.plan(proposal=requested);candidate=plans[0]
            except DispatchInfeasible:continue
            finally:self.network.spec=original_spec;self.network.smax=original_smax
            audit=self.network.audit(row,candidate['action'])
            if audit['ac_converged'] and audit['ac_violations']==0:chosen=candidate;break
        if chosen is None and self.emergency_enabled:
            from .emergency import emergency_plan
            try:chosen,emergency_meta=emergency_plan(self)
            except DispatchInfeasible:
                self.ac_rejected=True
                raise
        if chosen is None:
            self.ac_rejected=True
            raise DispatchInfeasible('AC 校核拒绝执行：有限重求解后无已验证可行动作；仿真停止，未虚构紧急后备')
        if emergency_meta:
            # 后备动作已验收；避免因服务违约再次触发失败的 MILP。原违约数仍完整记录。
            original_config=self.config
            self.config=copy.copy(original_config);self.config.safety=False
            try:obs,share,reward,done,info=super().step_physical(chosen)
            finally:self.config=original_config
        else:obs,share,reward,done,info=super().step_physical(chosen)
        info.update(emergency_meta)
        info.update(pre_shield_violations=pre,shield_l1_kw=float(abs(chosen['action']-requested).sum()),
            ac_safety_attempts=attempts,ac_safety_seconds=time.perf_counter()-start)
        return obs,share,reward,done,info
