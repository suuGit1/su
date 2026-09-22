"""含网损的并网点功率跟踪与有限点备用确认；不作全区间安全承诺。"""
import copy
from dataclasses import replace
import numpy as np
from vpp_mappo.optimization import DispatchInfeasible
from .checked_reserve import ac_economic_plan

TOLERANCE_KW = 0.1
VERSION = 'c3-ac-pcc-cost-carbon-sampled-reserve-v3'


def target_plan(core, target_kw, tolerance_kw=TOLERANCE_KW, max_iterations=12):
    """外层网损补偿，内层统一 MILP；只返回通过真实 AC 校核的动作。"""
    if not np.isfinite([target_kw, tolerance_kw]).all() or tolerance_kw <= 0 or max_iterations < 1:
        raise ValueError('PCC 目标、容差或迭代次数非法')
    original, smax = core.network.spec, core.network.smax
    errors = []
    # 修正等式中的线性目标，而不是把交流功率当成线性功率报告。
    for margin in (0., .002, .004, .006):
        linear_target = float(target_kw)
        for iteration in range(max_iterations):
            try:
                core.network.spec = replace(original, voltage_min=original.voltage_min+margin,
                                            voltage_max=original.voltage_max-margin)
                core.network.smax = smax * (1-5*margin)
                plans, meta = core.plan(grid_target=linear_target)
            except DispatchInfeasible:
                break
            finally:
                core.network.spec, core.network.smax = original, smax
            plan = plans[0]
            audit = core.network.audit(core.row(), plan['action'])
            if not audit['ac_converged']:
                break
            error = float(audit['ac_grid_kw'] - target_kw)
            errors.append(error)
            if abs(error) <= tolerance_kw and audit['ac_violations'] == 0 and core.check(core.row(), plan['action'], plan['ev_kw']) == 0:
                return plan, dict(**meta, pcc_target_kw=float(target_kw), pcc_actual_kw=audit['ac_grid_kw'],
                                 pcc_error_kw=error, pcc_tolerance_kw=tolerance_kw,
                                 pcc_iterations=iteration+1, pcc_margin=margin, ac_loss_kw=audit['ac_loss_kw'])
            linear_target -= error
    raise DispatchInfeasible('PCC 目标未通过功率误差与 AC 约束联合验收；最后误差='+str(errors[-1:] or None))


def checked_capacity(core, baseline_kw, upper_kw, iterations=5, fractions=(-1., -.5, 0., .5, 1.)):
    """对指定离散激活比例同时检查；有限搜索不等于最大能力或连续区间证明。"""
    if not np.isfinite([baseline_kw, upper_kw]).all() or upper_kw < 0 or type(iterations) is not int or iterations < 0:
        raise ValueError('PCC 备用搜索参数非法')
    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or not np.isfinite(fractions).all() or not {-1., 0., 1.}.issubset(set(fractions)) or np.any(abs(fractions)>1):
        raise ValueError('激活比例须位于[-1,1]且包含两个端点和零')
    def check(q):
        records = []
        try:
            for fraction in fractions:
                _, meta = target_plan(core, baseline_kw+fraction*q)
                records.append(dict(fraction=float(fraction), target_kw=meta['pcc_target_kw'],
                                    actual_kw=meta['pcc_actual_kw'], error_kw=meta['pcc_error_kw']))
            return records
        except DispatchInfeasible:
            return None
    zero = check(0.)
    if zero is None:
        return dict(kw=0., baseline_feasible=False, sampled_checked=False, samples=[])
    candidate = check(upper_kw)
    if candidate is not None:
        return dict(kw=float(upper_kw), baseline_feasible=True, sampled_checked=True, samples=candidate)
    low, high, records = 0., float(upper_kw), zero
    for _ in range(iterations):
        mid = (low+high)/2; candidate = check(mid)
        if candidate is None: high = mid
        else: low, records = mid, candidate
    return dict(kw=low, baseline_feasible=True, sampled_checked=True, samples=records)


def activate(core, baseline_kw, reserve_kw, fraction, duration_steps=1):
    """克隆后逐步以 AC PCC 功率验收，继续回放剩余服务任务。"""
    if not np.isfinite([baseline_kw, reserve_kw, fraction]).all() or reserve_kw < 0 or abs(fraction)>1:
        raise ValueError('激活参数非法')
    if type(duration_steps) is not int or not 1 <= duration_steps <= core.config.horizon-core.t:
        raise ValueError('激活持续时间超出剩余时域')
    e = copy.deepcopy(core); target = baseline_kw+fraction*reserve_kw
    result = dict(failed=False, fraction=fraction, target_kw=target, duration_steps=duration_steps, steps=[])
    try:
        for _ in range(duration_steps):
            plan, _ = target_plan(e, target)
            *_, info = e.step_physical(plan)
            error = abs(info['ac_grid_kw']-target) if info['ac_converged'] else None
            result['steps'].append(dict(pcc_kw=info['ac_grid_kw'], linear_grid_kw=info['grid_power_kw'],
                                        error_kw=error, ac_violations=info['ac_violations']))
            if error is None or error > TOLERANCE_KW or info['constraint_violations'] or info['ac_violations']:
                raise DispatchInfeasible('激活后的实际执行未满足 PCC 功率或约束要求')
        while e.t < e.config.horizon:
            *_, info = e.step_physical(ac_economic_plan(e))
            if info['constraint_violations'] or info['ac_violations']:
                raise DispatchInfeasible('后续轨迹未满足约束')
        result.update(ev_unmet_kwh=sum(max(0.,v) for v in e.remaining.values()), dr_backlog_kwh=e.backlog)
        if result['ev_unmet_kwh']>1e-5 or abs(e.backlog)>1e-5:
            raise DispatchInfeasible('激活后服务任务未完成')
    except DispatchInfeasible as exc:
        result.update(failed=True, error=str(exc))
    return result
