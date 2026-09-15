"""会话级 EV 与 DR 跨时段约束的共享 MILP；返回可逐步执行的分车计划。"""
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from .optimization import DispatchInfeasible


def solve_flex(base, spec, network, rows, start, soc, backlog, shifted, shed_used,
               sessions, remaining, dt, terminal_weight, time_limit=30., proposal=None,
               objective='economic', objective_steps=None):
    h=len(rows); n=len(sessions); stride=15+n; total=h*stride+1+(6 if proposal is not None else 0)
    if not h or objective not in ('economic','min_grid','max_grid'): raise ValueError('规划长度或目标不合法')
    window=h if objective_steps is None else min(h,objective_steps)
    if window<1:raise ValueError('优化窗口必须为正')
    c=np.zeros(total);lb=np.zeros(total);ub=np.full(total,np.inf);integ=np.zeros(total)
    ri=[];ci=[];vv=[];lower=[];upper=[]
    def con(coef,lo=-np.inf,hi=np.inf):
        r=len(lower)
        for j,v in coef.items():
            if v:ri.append(r);ci.append(j);vv.append(v)
        lower.append(lo);upper.append(hi)
    def action(t,k):
        b=t*stride
        return ({b+1:1,b:-1},{b+2:1},{b+3:1,b+4:-1},{b+5:1},{b+6:1},{b+7:1})[k]
    for t,row in enumerate(rows):
        b=t*stride; cap=base.capacities[0];p=base.power_max[0]
        ub[b]=ub[b+1]=p;ub[b+2]=spec.ev_station_kw
        ub[b+3]=spec.dr_shift_kw;ub[b+4]=spec.dr_repay_kw;ub[b+5]=spec.dr_shed_kw
        ub[b+6]=row['pv_kw'];ub[b+7]=row['wind_kw'];ub[b+8]=base.grid_import;ub[b+9]=base.grid_export
        lb[b+10]=base.soc_min[0];ub[b+10]=base.soc_max[0];ub[b+11]=spec.dr_backlog_kwh
        for j in (12,13,14):ub[b+j]=1;integ[b+j]=1
        con({b:1,b+12:-p},hi=0);con({b+1:1,b+12:p},hi=p)
        con({b+3:1,b+13:-spec.dr_shift_kw},hi=0)
        con({b+4:1,b+13:spec.dr_repay_kw},hi=spec.dr_repay_kw)
        con({b+8:1,b+14:-base.grid_import},hi=0)
        con({b+9:1,b+14:base.grid_export},hi=base.grid_export)
        con({b+3:1,b+5:1},hi=spec.dr_fraction*row['load_kw'])
        eq={b+10:1,b:-dt*base.efficiency/cap,b+1:dt/(base.efficiency*cap)}
        if t:eq[b-stride+10]=-1
        con(eq,soc if t==0 else 0,soc if t==0 else 0)
        eq={b+11:1,b+3:-dt,b+4:dt}
        if t:eq[b-stride+11]=-1
        con(eq,backlog if t==0 else 0,backlog if t==0 else 0)
        ev={b+2:1}
        for i,s in enumerate(sessions):
            j=b+15+i;ev[j]=-1
            ub[j]=s['max_kw'] if s['arrival_step']<=start+t<s['departure_step'] else 0
        con(ev,0,0)
        net=row['load_kw']-row['pv_kw']-row['wind_kw']
        con({b+8:1,b+9:-1,b:-1,b+1:1,b+2:-1,b+3:1,b+4:-1,b+5:1,b+6:-1,b+7:-1},net,net)
        mat,lo,hi=network.affine(row)
        for coeff,l,u in zip(mat,lo,hi):
            eq={}
            for k,value in enumerate(coeff):
                for j,v in action(t,k).items():eq[j]=eq.get(j,0)+value*v
            con(eq,l,u)
        if t<window:
            for j in (0,1):c[b+j]=dt*base.degradation
            c[b+3]=c[b+4]=dt*spec.shift_cost;c[b+5]=dt*spec.shed_cost
            c[b+6]=c[b+7]=dt*spec.curtail_cost
            c[b+8]=dt*row['price'];c[b+9]=-dt*row['price']*base.sell_ratio
    for i,s in enumerate(sessions):
        # 需求为充电桩侧电量，不虚构初始 SOC，不允许未接入充电或 V2G。
        need=remaining[s['id']]
        con({t*stride+15+i:dt for t in range(h)},need,need)
    con({t*stride+3:dt for t in range(h)},hi=spec.dr_shift_budget_kwh-shifted)
    con({t*stride+5:dt for t in range(h)},hi=spec.dr_shed_budget_kwh-shed_used)
    con({(h-1)*stride+11:1},0,0)
    con({(window-1)*stride+10:1,h*stride:1},lo=base.target_soc[0]);c[h*stride]=terminal_weight
    if proposal is not None:
        c[:]=0
        scale=[base.power_max[0],spec.ev_station_kw,max(spec.dr_shift_kw,spec.dr_repay_kw),spec.dr_shed_kw,max(rows[0]['pv_kw'],1),max(rows[0]['wind_kw'],1)]
        for k in range(6):
            j=h*stride+1+k;c[j]=1/max(scale[k],1)
            eq=action(0,k);eq[j]=-1;con(eq,hi=proposal[k])
            eq={i:-v for i,v in action(0,k).items()};eq[j]=-1;con(eq,hi=-proposal[k])
    elif objective!='economic':
        c[:]=0;sign=1 if objective=='min_grid' else -1;c[8]=sign;c[9]=-sign
    matrix=coo_matrix((vv,(ri,ci)),shape=(len(lower),total)).tocsc()
    began=time.perf_counter()
    result=milp(c,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(matrix,lower,upper),options={'time_limit':time_limit,'mip_rel_gap':1e-6})
    seconds=time.perf_counter()-began
    if result.x is None or result.status not in (0,1):raise DispatchInfeasible(f'会话/DR 调度不可行或无可执行解：{result.message}')
    x=result.x; ax=matrix@x
    residual=max(float(np.max(np.maximum(np.asarray(lower)-ax,0))),float(np.max(np.maximum(ax-np.asarray(upper),0))),float(np.max(np.maximum(lb-x,0))),float(np.max(np.maximum(x-ub,0))),float(np.max(np.abs(x[integ==1]-np.round(x[integ==1])))))
    if not np.isfinite(x).all() or residual>1e-5:raise DispatchInfeasible('求解解未通过独立残差检查')
    plan=[]
    for t in range(h):
        a=[sum(x[j]*v for j,v in action(t,k).items()) for k in range(6)]
        plan.append({'action':np.asarray(a),'ev_kw':{s['id']:float(x[t*stride+15+i]) for i,s in enumerate(sessions)}})
    return plan,dict(solver_status=int(result.status),solver_optimal=result.status==0,solver_objective=float(result.fun),solver_bound=float(result.mip_dual_bound),solver_gap=float(result.mip_gap),solver_seconds=seconds,solver_residual=residual)
