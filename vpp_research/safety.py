"""区间盒内的单步线性网络/SOC 保护；不声称全时域或 AC 鲁棒保证。"""
import itertools
import time
import numpy as np
from scipy.optimize import linprog


def guard(action,observation,dispatch,flex,network,dt,horizon):
    began=time.perf_counter();x=np.asarray(observation);a=np.asarray(action);w=x[54:63]
    center=np.array([x[9]*5000,x[11]*1000,x[12]*1000]);width=np.array([w[5]*5000,w[7]*1000,w[8]*1000])
    lows=np.maximum(0,center-width);highs=np.maximum(0,center+width)
    matrix=[];upper=[]
    def add(row,bound):matrix.append(np.asarray(row));upper.append(float(bound))
    for values in itertools.product(*zip(lows,highs)):
        row=dict(zip(('load_kw','pv_kw','wind_kw'),values));row['price']=x[10]*.3
        m,lo,hi=network.affine(row)
        for r,l,u in zip(m,lo,hi):add(r,u);add(-r,-l)
        net=values[0]-values[1]-values[2];g=np.array([-1,1,-1,-1,1,1])
        add(g,dispatch.grid_import-net);add(-g,dispatch.grid_export+net)
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
        result=linprog(np.r_[np.zeros(6),1/scales],A_ub=amat,b_ub=rhs,bounds=bounds+[(0,None)]*6,method='highs')
        status=str(result.message)
    ok=result is not None and result.success
    candidate=result.x[:6] if ok else a.copy()
    def certified(executed):
        return bool(ok and np.max(m@executed-u)<=1e-5 and all(lo-1e-5<=v<=hi+1e-5 for v,(lo,hi) in zip(executed,bounds)))
    return candidate,dict(guard_feasible=bool(ok),guard_status=status,
        guard_inconsistent_dimensions=[i for i,(lo,hi) in enumerate(bounds) if lo>hi],
        guard_action_bounds=[list(b) for b in bounds],guard_seconds=time.perf_counter()-began),certified
