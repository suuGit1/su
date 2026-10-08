"""区间盒内的单步线性网络/SOC 保护；不声称全时域或 AC 鲁棒保证。"""
import itertools
import time
import numpy as np
from scipy.optimize import linprog


def guard(action,observation,dispatch,flex,network,dt,horizon,diagnostics=False,time_limit=30.):
    began=time.perf_counter();x=np.asarray(observation);a=np.asarray(action);w=x[54:63]
    center=np.array([x[9]*5000,x[11]*1000,x[12]*1000]);width=np.array([w[5]*5000,w[7]*1000,w[8]*1000])
    lows=np.maximum(0,center-width);highs=np.maximum(0,center+width)
    matrix=[];upper=[];labels=[]
    def add(row,bound,label='resource'):
        matrix.append(np.asarray(row));upper.append(float(bound));labels.append(label)
    for corner,values in enumerate(itertools.product(*zip(lows,highs))):
        row=dict(zip(('load_kw','pv_kw','wind_kw'),values));row['price']=x[10]*.3
        m,lo,hi=network.affine(row)
        for index,(r,l,u) in enumerate(zip(m,lo,hi)):
            nb=len(network.net.bus);nl=len(network.lines)
            kind='voltage' if index<nb else 'active_flow' if index<nb+nl else 'reactive_flow'
            add(r,u,f'corner_{corner}_{kind}_{index}_upper');add(-r,-l,f'corner_{corner}_{kind}_{index}_lower')
        net=values[0]-values[1]-values[2];g=np.array([-1,1,-1,-1,1,1])
        add(g,dispatch.grid_import-net,f'corner_{corner}_grid_import');add(-g,dispatch.grid_export+net,f'corner_{corner}_grid_export')
    w=w.copy()
    if flex.dr_shift_kw==0 and flex.dr_repay_kw==0:w[2:4]=0.
    if flex.dr_shed_kw==0:w[4]=0.
    # 与已知物理支撑集取交集，不把不可能的SOC状态加入不确定集。
    low_soc=max(dispatch.soc_min[0],x[1]-w[0]);high_soc=min(dispatch.soc_max[0],x[1]+w[0]);cap=dispatch.capacities[0];eta=dispatch.efficiency
    b_low=max(0,(x[6]-w[2])*max(1,flex.dr_backlog_kwh));b_high=(x[6]+w[2])*max(1,flex.dr_backlog_kwh)
    steps=max(0,horizon-round(x[0]*horizon)-1)
    bounds=[(max(-dispatch.power_max[0],-(dispatch.soc_max[0]-high_soc)*cap/(eta*dt)),
             min(dispatch.power_max[0],(low_soc-dispatch.soc_min[0])*cap*eta/dt)),
            (0,flex.ev_station_kw),
            (max(-flex.dr_repay_kw,-b_low/dt),min(flex.dr_shift_kw,(flex.dr_backlog_kwh-b_high)/dt,
             (flex.dr_shift_budget_kwh-(x[7]+w[3])*max(1,flex.dr_shift_budget_kwh))/dt,flex.dr_repay_kw*steps-b_high/dt)),
            (0,min(flex.dr_shed_kw,(flex.dr_shed_budget_kwh-(x[8]+w[4])*max(1,flex.dr_shed_budget_kwh))/dt)),
            (0,lows[1]),(0,lows[2])]
    add([0,0,0,1,0,0],flex.dr_fraction*lows[0]);add([0,0,1,1,0,0],flex.dr_fraction*lows[0])
    m=np.asarray(matrix);u=np.asarray(upper)
    status='inconsistent_interval';result=None
    support_valid=low_soc<=high_soc
    if support_valid and all(lo<=hi for lo,hi in bounds):
        eye=np.eye(6);amat=np.block([[m,np.zeros_like(m)],[eye,-eye],[-eye,-eye]])
        rhs=np.r_[u,a,-a]
        scales=np.array([dispatch.power_max[0],flex.ev_station_kw,max(1,flex.dr_shift_kw,flex.dr_repay_kw),max(1,flex.dr_shed_kw),max(1,center[1]),max(1,center[2])])
        result=linprog(np.r_[np.zeros(6),1/scales],A_ub=amat,b_ub=rhs,bounds=bounds+[(0,None)]*6,method='highs',options={'time_limit':time_limit})
        status=str(result.message)
    ok=result is not None and result.success
    candidate=result.x[:6] if ok else a.copy()
    def certified(executed):
        return bool(ok and np.max(m@executed-u)<=1e-5 and all(lo-1e-5<=v<=hi+1e-5 for v,(lo,hi) in zip(executed,bounds)))
    certified.constraints=(m,u,bounds) if ok else None
    extra={}
    if diagnostics:
        reason='certified_linear_box' if ok else 'inconsistent_support' if not support_valid else 'inconsistent_action_bounds' if any(lo>hi for lo,hi in bounds) else 'robust_polytope_infeasible' if result is not None and result.status==2 else 'solver_failure'
        extra=dict(guard_reason=reason,guard_scope='单步线性网络与SOC/DR约束盒；不含连续AC鲁棒或EV期限证明',guard_box_low=lows.tolist(),guard_box_high=highs.tolist(),guard_soc_interval=[float(low_soc),float(high_soc)],guard_solver_status=int(result.status) if result is not None else None)
        if not ok and support_valid and all(lo<=hi for lo,hi in bounds):
            # 松弛解只解释冲突，绝不用于控制或缩窄校准区间。
            scale=np.maximum(1,np.maximum(abs(u),np.sum(abs(m),axis=1)))
            remaining=max(.001,time_limit-(time.perf_counter()-began))
            relaxed=linprog(np.r_[np.zeros(6),1.],A_ub=np.c_[m,-scale],b_ub=u,bounds=bounds+[(0,None)],method='highs',options={'time_limit':remaining})
            if relaxed.success:
                slack=np.maximum(m@relaxed.x[:6]-u,0)/scale
                extra.update(guard_diagnostic_relaxation=float(relaxed.x[-1]),guard_conflicting_constraints=[dict(constraint=labels[i],normalized_slack=float(slack[i])) for i in np.flatnonzero(slack>1e-7)],guard_relaxation_used_for_execution=False)
    return candidate,dict(**extra,guard_feasible=bool(ok),guard_status=status,
        guard_inconsistent_dimensions=[i for i,(lo,hi) in enumerate(bounds) if lo>hi],
        guard_action_bounds=[list(b) for b in bounds],guard_seconds=time.perf_counter()-began),certified
