"""与策略观察等价的聚合 MPC；不能访问环境、DT 缓存或真实单车记录。"""
import time
import numpy as np
from .flex_resources import FlexNetwork
from .flex_optimization import solve_flex
from .optimization import DispatchInfeasible

CONTRACT_VERSION='public-observation-aggregate-mpc-v1'


class ObservedMPC:
    def __init__(self,config,dispatch,flex,lookahead=6):
        if lookahead<1:raise ValueError('MPC 窗口必须为正')
        self.config=config;self.dispatch=dispatch;self.flex=flex;self.lookahead=lookahead
        self.network=FlexNetwork(dispatch)

    def propose(self,observation):
        x=np.asarray(observation,dtype=float)
        if x.shape!=(54,) or not np.isfinite(x).all():raise ValueError('MPC 要求有限的 54 维公开观察')
        c=self.config;step=round(x[0]*c.horizon);h=c.horizon-step
        if h<1:raise ValueError('MPC 已到终端')
        row=dict(load_kw=max(0,x[9]*5000),price=max(0,x[10]*.3),pv_kw=max(0,x[11]*1000),wind_kw=max(0,x[12]*1000))
        need=max(0,x[3]*1000);deadline=max(1,min(h,round(x[5]*c.horizon)))
        # 与 54 维策略输入保持一致：不额外读取 DT 内部逐车信息。
        # 将全部已知聚合需求放在最近期限，属于保守的期限聚合近似。
        sessions=[];remaining={}
        if need>1e-7:
            sessions=[dict(id='observed_aggregate',arrival_step=step,departure_step=step+deadline,energy_kwh=need,max_kw=self.flex.ev_station_kw)]
            remaining={'observed_aggregate':need}
        started=time.perf_counter()
        try:
            plans,meta=solve_flex(self.dispatch,self.flex,self.network,[row.copy() for _ in range(h)],step,
                float(x[1]),max(0,x[6]*max(1,self.flex.dr_backlog_kwh)),
                max(0,x[7]*max(1,self.flex.dr_shift_budget_kwh)),max(0,x[8]*max(1,self.flex.dr_shed_budget_kwh)),
                sessions,remaining,c.dt_hours,c.terminal_soc_penalty,c.solver_time_limit,
                objective_steps=min(self.lookahead,h))
            a=plans[0]['action'];failed=False;reason=''
        except DispatchInfeasible as exc:
            # 不修改需求或伪造可行解；明确记录失败，提交零动作给共享本地安全执行器。
            a=np.zeros(6);meta=dict(solver_optimal=False,solver_seconds=None,solver_objective=None)
            failed=True;reason=str(exc)
        return a,dict(mpc_contract=CONTRACT_VERSION,mpc_failed=failed,mpc_failure_reason=reason,
            mpc_optimal=meta['solver_optimal'],mpc_solver_seconds=time.perf_counter()-started,
            mpc_predicted_objective=meta['solver_objective'],mpc_ev_model='aggregate_nearest_deadline_station_limit')
