"""现场AC验收、区间约束保留和有界后备；无硬可行动作则拒绝状态推进。"""
from dataclasses import replace
import copy
import time
import numpy as np
from vpp_mappo.flex_environment import FlexVPPAdapter
from vpp_mappo.optimization import DispatchInfeasible


class ACCheckedFlex(FlexVPPAdapter):
    def __init__(self,config,csv_path=None,emergency=False):
        super().__init__(config,csv_path);self.emergency_enabled=bool(emergency)

    def reset(self,scenario_seed,profile_index=0):
        self.ac_rejected=False;self.guard_constraints=None
        return super().reset(scenario_seed,profile_index)

    def step_physical(self,control):
        if self.ac_rejected:raise DispatchInfeasible('AC已拒绝本回合，必须reset后再运行')
        if self.t>=self.config.horizon:raise RuntimeError('回合已结束')
        requested=np.asarray(control['action'] if isinstance(control,dict) else control,dtype=float)
        if requested.shape!=(6,) or not np.isfinite(requested).all():raise ValueError('AC候选动作非法')
        row=self.row();pre=self.check(row,requested,self._allocate(requested[1]));started=time.perf_counter()
        spec,smax=self.network.spec,self.network.smax
        protected=self.guard_constraints;chosen=None;failures=[];emergency_meta={};attempts=0
        original_config=self.config
        def valid(candidate):
            action=np.asarray(candidate['action']);allocation=candidate['ev_kw']
            if not set(allocation)<=set(self.remaining) or not all(np.isfinite(v) for v in allocation.values()):return False
            if self.check(row,action,allocation):return False
            if protected is not None:
                m,u,bounds=protected
                if np.max(m@action-u)>1e-5 or any(not lo-1e-5<=v<=hi+1e-5 for v,(lo,hi) in zip(action,bounds)):return False
            audit=self.network.audit(row,action)
            if not audit['ac_converged'] or audit['ac_violations']:
                failures.append(dict(reason='ac_violation' if audit['ac_converged'] else 'ac_nonconvergence',audit=audit));return False
            return True
        try:
            if isinstance(control,dict) and valid(control):chosen=control
            for margin in (() if chosen is not None else (0.,.002,.004,.006)):
                remaining=original_config.solver_time_limit-(time.perf_counter()-started)
                if original_config.safety_revision==1 and remaining<=0:
                    failures.append(dict(reason='timeout'));break
                attempts+=1
                try:
                    self.network.spec=replace(spec,voltage_min=spec.voltage_min+margin,voltage_max=spec.voltage_max-margin)
                    self.network.smax=smax*(1-margin*5)
                    self.config=copy.copy(original_config)
                    if original_config.safety_revision==1:self.config.solver_time_limit=max(.001,remaining)
                    plans,_=self.plan(proposal=requested)
                except DispatchInfeasible as exc:
                    failures.append(dict(reason=exc.reason,message=str(exc),details=exc.details));continue
                finally:
                    self.config=original_config;self.network.spec=spec;self.network.smax=smax
                if valid(plans[0]):chosen=plans[0];break
            if chosen is None and protected is not None:
                failures.append(dict(reason='interval_execution_infeasible'));self.guard_constraints=None
            if chosen is None and self.emergency_enabled:
                from .emergency import emergency_plan
                chosen,emergency_meta=emergency_plan(self)
            if chosen is None:raise DispatchInfeasible('没有经过硬约束和AC验收的动作，拒绝状态推进',reason='no_safe_action')
            hard_verified=self.check(row,chosen['action'],chosen['ev_kw'],include_service=False)==0
            if not hard_verified:raise DispatchInfeasible('后备硬约束复核失败',reason='no_safe_action')
            if emergency_meta:
                self.config=copy.copy(original_config);self.config.safety=False
            result=super().step_physical(chosen);info=result[-1];info.update(emergency_meta)
            info.update(pre_shield_violations=pre,shield_l1_kw=float(abs(np.asarray(chosen['action'])-requested).sum()),ac_safety_attempts=attempts,ac_safety_seconds=time.perf_counter()-started)
            if original_config.safety_revision==1:
                info.update(safety_failures=failures,safety_hard_verified=hard_verified and info['ac_converged'] and info['ac_violations']==0,safety_mode='emergency_degraded' if emergency_meta else 'interval_projected' if protected is not None else 'nominal_local')
            return result
        except DispatchInfeasible as exc:
            self.ac_rejected=True
            exc.details.update(safety_failures=failures,state_step=self.t)
            raise
        finally:
            self.config=original_config;self.network.spec=spec;self.network.smax=smax;self.guard_constraints=None
