"""HiGHS MILP：互斥充放电/购售电、SOC、节点电压与支路约束。"""
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


class DispatchInfeasible(RuntimeError):
    pass


def solve_dispatch(spec, network, rows, soc, dt, terminal_weight, time_limit=30, proposal=None):
    # 每步 12 个变量：充电2、放电2、DR、购电、售电、储能模式2、购电模式、SOC2。
    h = len(rows); size = 12*h+2+(3 if proposal is not None else 0)
    c = np.zeros(size); lb = np.zeros(size); ub = np.full(size, np.inf); integer = np.zeros(size)
    indices, columns, values, lows, highs = [], [], [], [], []
    def constraint(coeff, low=-np.inf, high=np.inf):
        rid = len(lows)
        for j, v in coeff.items():
            if v: indices.append(rid); columns.append(j); values.append(v)
        lows.append(low); highs.append(high)
    def action_coeff(t, k):
        b = t*12
        return {b+k+2:1, b+k:-1} if k < 2 else {b+4:1}
    def combine(t, coeff):
        result = {}
        for k in range(3):
            for j, v in action_coeff(t, k).items(): result[j] = result.get(j, 0)+coeff[k]*v
        return result
    for t, row in enumerate(rows):
        b = 12*t
        for k in range(2):
            cap = spec.capacities[k]; pmax = spec.power_max[k]
            ub[b+k] = ub[b+k+2] = pmax
            ub[b+7+k] = 1; integer[b+7+k] = 1
            lb[b+10+k], ub[b+10+k] = spec.soc_min[k], spec.soc_max[k]
            constraint({b+k:1,b+7+k:-pmax}, high=0)
            constraint({b+k+2:1,b+7+k:pmax}, high=pmax)
            eq = {b+10+k:1,b+k:-dt*spec.efficiency/cap,b+k+2:dt/(spec.efficiency*cap)}
            if t: eq[b-12+10+k] = -1
            rhs = soc[k] if t == 0 else 0
            constraint(eq, rhs, rhs)
            c[b+k] = c[b+k+2] = dt*spec.degradation
        ub[b+4] = min(spec.dr_max, row['load_kw'])
        ub[b+5], ub[b+6] = spec.grid_import, spec.grid_export
        integer[b+9] = 1; ub[b+9] = 1
        constraint({b+5:1,b+9:-spec.grid_import}, high=0)
        constraint({b+6:1,b+9:spec.grid_export}, high=spec.grid_export)
        net = row['load_kw']-row['pv_kw']-row['wind_kw']
        constraint({b+5:1,b+6:-1,b:-1,b+1:-1,b+2:1,b+3:1,b+4:1}, net, net)
        c[b+4], c[b+5], c[b+6] = dt*spec.dr_cost, dt*row['price'], -dt*row['price']*spec.sell_ratio
        a, low, high = network.affine(row)
        for coef, l, u in zip(a, low, high): constraint(combine(t, coef), l, u)
    for k in range(2):
        c[12*h+k] = terminal_weight
        constraint({12*(h-1)+10+k:1,12*h+k:1}, low=spec.target_soc[k])
    if proposal is not None:
        if h != 1: raise ValueError('安全投影只允许单步')
        c[:] = 0
        for k in range(3):
            j = 12*h+2+k; c[j] = 1/(spec.power_max[k] if k < 2 else spec.dr_max)
            eq = action_coeff(0, k); eq[j] = -1
            constraint(eq, high=proposal[k])
            eq = {i:-v for i,v in action_coeff(0,k).items()}; eq[j] = -1
            constraint(eq, high=-proposal[k])
    matrix = coo_matrix((values,(indices,columns)), shape=(len(lows),size)).tocsc()
    start = time.perf_counter()
    result = milp(c, integrality=integer, bounds=Bounds(lb,ub), constraints=LinearConstraint(matrix,lows,highs),
                  options={'time_limit':time_limit,'mip_rel_gap':1e-6})
    elapsed = time.perf_counter()-start
    if result.x is None or result.status not in (0,1):
        raise DispatchInfeasible(f'MILP 无可执行解，status={result.status}: {result.message}')
    x = result.x; ax = matrix@x
    residual = max(float(np.max(np.maximum(np.asarray(lows)-ax,0))),float(np.max(np.maximum(ax-np.asarray(highs),0))),
                   float(np.max(np.maximum(lb-x,0))),float(np.max(np.maximum(x-ub,0))),
                   float(np.max(np.abs(x[integer==1]-np.round(x[integer==1])))))
    if residual > 1e-5: raise DispatchInfeasible(f'求解器返回解未通过可行性检查：{residual}')
    actions = np.array([[x[t*12+2]-x[t*12],x[t*12+3]-x[t*12+1],x[t*12+4]] for t in range(h)])
    meta = dict(solver_status=int(result.status), solver_optimal=result.status==0, solver_objective=float(result.fun),
                solver_bound=float(result.mip_dual_bound), solver_gap=float(result.mip_gap), solver_seconds=elapsed,
                solver_residual=residual)
    return actions, meta
