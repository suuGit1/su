"""无 MILP 的有限候选现场后备；服务降级显式记账，找不到硬安全动作则拒绝执行。"""
import itertools
import numpy as np
from vpp_mappo.optimization import DispatchInfeasible


def emergency_plan(core):
    row, dt, spec, flex = core.row(), core.config.dt_hours, core.spec, core.flex
    # 只使用已接入车辆与当前现场状态，不读取未来实际曲线。
    ev_max = min(flex.ev_station_kw, sum(min(s['max_kw'], max(0,core.remaining[s['id']])/dt) for s in core.active()))
    repay = min(flex.dr_repay_kw, max(0,core.backlog)/dt)
    discharge = min(spec.power_max[0], max(0,core.soc[0]-spec.soc_min[0])*spec.capacities[0]*spec.efficiency/dt)
    charge = min(spec.power_max[0], max(0,spec.soc_max[0]-core.soc[0])*spec.capacities[0]/(spec.efficiency*dt))
    shed = min(flex.dr_shed_kw, max(0,flex.dr_shed_budget_kwh-core.shed_used)/dt, flex.dr_fraction*row['load_kw'])
    candidates = []
    for e, ev, shift, dr, curtail in itertools.product((0., discharge, -charge), (ev_max, 0.), (-repay, 0.), (0., shed), (0., .5, 1.)):
        a = np.array([e, ev, shift, dr, curtail*row['pv_kw'], curtail*row['wind_kw']])
        allocation = core._allocate(ev)
        if core.check(row, a, allocation, include_service=False): continue
        # 优先避免服务可达性破坏，再减少未充需求/DR积压/削减；不声称经济最优。
        service = core.check(row, a, allocation)
        score = (service, ev_max-ev, core.backlog+shift*dt, dr, curtail, abs(e))
        candidates.append((score, a, allocation))
    attempts = 0
    for score, action, allocation in sorted(candidates, key=lambda x:x[0]):
        attempts += 1; audit = core.network.audit(row, action)
        if audit['ac_converged'] and audit['ac_violations']==0:
            return dict(action=action, ev_kw=allocation), dict(emergency=True, emergency_ac_attempts=attempts,
                emergency_service_degraded=bool(score[0]), emergency_service_violations=int(score[0]),
                emergency_hard_violations=0, emergency_policy='finite-local-candidates-v1')
    raise DispatchInfeasible('紧急后备候选中没有通过硬约束与 AC 校核的动作；保持仿真状态并拒绝执行')
