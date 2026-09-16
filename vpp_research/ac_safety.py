"""现场 AC 校核与有限重求解；找不到可验证动作时终止仿真，不伪造后备保证。"""
from dataclasses import replace
import time
import numpy as np
from vpp_mappo.flex_environment import FlexVPPAdapter
from vpp_mappo.optimization import DispatchInfeasible


class ACCheckedFlex(FlexVPPAdapter):
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
        chosen=None;attempts=0
        # 仅改变优化器的裕度，AC 验收始终使用原始物理边界。
        for margin in (0.,.002,.004,.006):
            attempts+=1
            try:
                self.network.spec=replace(original_spec,voltage_min=original_spec.voltage_min+margin,voltage_max=original_spec.voltage_max-margin)
                self.network.smax=original_smax*(1-margin*5)
                plans,_=self.plan(proposal=requested);candidate=plans[0]
            except DispatchInfeasible:continue
            finally:self.network.spec=original_spec;self.network.smax=original_smax
            audit=self.network.audit(row,candidate['action'])
            if audit['ac_converged'] and audit['ac_violations']==0:chosen=candidate;break
        if chosen is None:
            self.ac_rejected=True
            raise DispatchInfeasible('AC 校核拒绝执行：有限重求解后无已验证可行动作；仿真停止，未虚构紧急后备')
        obs,share,reward,done,info=super().step_physical(chosen)
        info.update(pre_shield_violations=pre,shield_l1_kw=float(abs(chosen['action']-requested).sum()),
            ac_safety_attempts=attempts,ac_safety_seconds=time.perf_counter()-start)
        return obs,share,reward,done,info
