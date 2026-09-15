"""三目标口径：经济、运行碳排与恒定工况线性可交付备用。"""
import numpy as np

OBJECTIVE_NAMES = ('economic_return', 'carbon_return', 'flexibility_return')
CONTRACT_VERSION = 'economic-carbon-reserve-v1'


def carbon_kg(grid_kw, dt_hours, carbon_g_per_kwh):
    if not np.isfinite([grid_kw, dt_hours, carbon_g_per_kwh]).all() or dt_hours <= 0 or carbon_g_per_kwh < 0:
        raise ValueError('碳核算参数无效')
    # 核算边界为主网进口运行排放；出口不抵扣，储能放电不重复计入。
    return max(grid_kw, 0.0)*dt_hours*carbon_g_per_kwh/1000


def reserve_envelope(spec, network, row, post_soc, baseline_grid_kw, sustain_hours, time_limit):
    from .optimization import solve_dispatch
    limits = []
    seconds = 0.0
    for objective in ('min_grid', 'max_grid'):
        actions, meta = solve_dispatch(spec, network, [row], post_soc, sustain_hours, 0.0,
                                       time_limit, objective=objective)
        if not meta['solver_optimal']:
            raise RuntimeError('备用包络未求至最优，不能把可行样本作为真实边界')
        limits.append(row['load_kw']-row['pv_kw']-row['wind_kw']-float(actions[0].sum()))
        seconds += meta['solver_seconds']
    lower, upper = limits
    up = max(0.0, baseline_grid_kw-lower)
    down = max(0.0, upper-baseline_grid_kw)
    baseline_feasible = lower-1e-5 <= baseline_grid_kw <= upper+1e-5
    # 当前基准功率若无法由执行后状态持续维持，则不宣称可交付对称备用。
    symmetric = min(up, down) if baseline_feasible else 0.0
    return dict(reserve_up_kw=up, reserve_down_kw=down, reserve_symmetric_kw=symmetric,
                reserve_baseline_feasible=bool(baseline_feasible), reserve_min_grid_kw=lower,
                reserve_max_grid_kw=upper, reserve_solver_seconds=seconds)


def step_metrics(config, spec, network, row, post_soc, info):
    factor = row['carbon_g_per_kwh']
    if info['constraint_violations']:
        reserve = dict(reserve_up_kw=0.0,reserve_down_kw=0.0,reserve_symmetric_kw=0.0,
            reserve_baseline_feasible=False,reserve_min_grid_kw=None,reserve_max_grid_kw=None,
            reserve_solver_seconds=0.0,reserve_valid=False)
    else:
        reserve = reserve_envelope(spec, network, row, post_soc, info['grid_power_kw'],
                                   config.reserve_hours, config.solver_time_limit)
        reserve['reserve_valid'] = True
    kg = carbon_kg(info['grid_power_kw'], config.dt_hours, factor)
    flexibility = reserve['reserve_symmetric_kw']*config.dt_hours
    raw = np.array([-(info['cost']+info['terminal_penalty']), -kg, flexibility])
    normalized = raw/np.asarray(config.objective_scales)
    result = dict(carbon_g_per_kwh=factor, carbon_kg=kg,
        ac_carbon_kg=carbon_kg(info['ac_grid_kw'],config.dt_hours,factor) if info['ac_converged'] else None,
        flexibility_kw_hours=flexibility, **reserve)
    result.update(dict(zip(OBJECTIVE_NAMES, map(float, normalized))))
    return result


def aggregate_metrics(rows):
    result = {k:float(sum(r[k] for r in rows)) for k in (*OBJECTIVE_NAMES,'carbon_kg','flexibility_kw_hours')}
    result['ac_carbon_kg'] = sum(r['ac_carbon_kg'] for r in rows) if all(r['ac_carbon_kg'] is not None for r in rows) else None
    result['reserve_mean_kw'] = float(np.mean([r['reserve_symmetric_kw'] for r in rows]))
    result['reserve_solver_seconds'] = sum(r['reserve_solver_seconds'] for r in rows)
    result['reserve_invalid_steps'] = sum(not r['reserve_valid'] for r in rows)
    return result
