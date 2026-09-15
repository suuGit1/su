"""六资源智能体：ESS、会话 EV、负荷转移、负荷削减、弃光、弃风。"""
from dataclasses import asdict
import numpy as np
from gymnasium.spaces import Box
from .grid_environment import GridVPPAdapter
from .dispatch import DispatchSpec
from .flex_resources import FlexSpec, FlexNetwork, read_bundle, validate_sessions, grid_power
from .flex_optimization import solve_flex

CONTRACT_VERSION='sessions-dr-curtail-v1'


class FlexVPPAdapter:
    num_agents=6
    agent_names=['ess','ev','dr_shift','dr_shed','pv','wind']
    action_space=Box(-np.inf,np.inf,(1,),dtype=np.float32)

    def __init__(self, config, csv_path=None):
        # 复用原来的数据校验，物理状态与求解器使用新的版本。
        self.legacy=GridVPPAdapter(config,csv_path)
        self.config=config;self.data=self.legacy.data;self.spec=self.legacy.spec
        self.flex=FlexSpec.load(getattr(config,'_flex_record',None) or config.flex_spec)
        self.network=FlexNetwork(self.spec)
        self.bundle=getattr(config,'_ev_bundle',None) or (read_bundle(config.ev_sessions_path) if config.ev_sessions_path else None)
        if self.data and self.bundle is None:raise ValueError('真实日曲线必须配套 EV 会话 JSON；不能自动生成并冒称真实会话')
        if self.bundle and (self.bundle.get('horizon')!=config.horizon or self.bundle.get('dt_hours')!=config.dt_hours):raise ValueError('EV 会话的时域/时间步长与配置不同')
        d=18+int(config.metrics_enabled)
        self.observation_space=Box(-np.inf,np.inf,(d,),dtype=np.float32)
        self.share_observation_space=Box(-np.inf,np.inf,(d*6,),dtype=np.float32)

    def reset(self, scenario_seed, profile_index=0):
        self.legacy.reset(scenario_seed,profile_index);self.profiles=self.legacy.profiles
        self.t=0;self.soc=np.array([self.spec.initial_soc[0],0.0]);self.backlog=0.;self.shifted=0.;self.shed_used=0.
        key=self.data.scenario_names[profile_index%len(self.data.profiles)] if self.data else f'synthetic:{scenario_seed}'
        if self.bundle:
            if key not in self.bundle['scenarios']:raise ValueError('EV 会话文件缺少场景 '+key)
            records=self.bundle['scenarios'][key]
        else:
            h=self.config.horizon;dt=self.config.dt_hours
            records=[dict(id='demo_0',arrival_step=0,departure_step=h,energy_kwh=min(12.,7.*dt*h),max_kw=11.),
                     dict(id='demo_1',arrival_step=h//2,departure_step=h,energy_kwh=min(8.,5.*dt*(h-h//2)),max_kw=7.)]
        self.sessions=validate_sessions(records,self.config.horizon,self.config.dt_hours)
        self.remaining={s['id']:float(s['energy_kwh']) for s in self.sessions}
        return self.encode()

    def row(self,t=None):return {k:float(v[self.t if t is None else t]) for k,v in self.profiles.items()}
    def active(self):return [s for s in self.sessions if s['arrival_step']<=self.t<s['departure_step']]

    def encode(self):
        if self.t>=self.config.horizon:return np.zeros((6,*self.observation_space.shape),np.float32),np.zeros((6,*self.share_observation_space.shape),np.float32)
        r=self.row();active=self.active();need=sum(max(0,self.remaining[s['id']]) for s in active)
        common=[self.t/self.config.horizon,r['price']/.3,(r['load_kw']-r['pv_kw']-r['wind_kw'])/5000,r['pv_kw']/1000,r['wind_kw']/1000,
                self.backlog/max(self.flex.dr_backlog_kwh,1),self.shifted/max(self.flex.dr_shift_budget_kwh,1),self.shed_used/max(self.flex.dr_shed_budget_kwh,1),len(active)/100,need/1000]
        if self.config.metrics_enabled:common.append(r['carbon_g_per_kwh']/1000)
        deadline=min((s['departure_step']-self.t for s in active),default=0)/self.config.horizon
        own=[(self.soc[0],self.spec.target_soc[0]),(need/1000,deadline),(self.backlog/max(self.flex.dr_backlog_kwh,1),0),
             (self.shed_used/max(self.flex.dr_shed_budget_kwh,1),0),(r['pv_kw']/1000,0),(r['wind_kw']/1000,0)]
        obs=np.asarray([common+list(own[i])+np.eye(6)[i].tolist() for i in range(6)],np.float32)
        return obs,np.repeat(obs.reshape(1,-1),6,axis=0)

    def plan(self, oracle=False, lookahead=6, proposal=None, objective='economic', fixed_row=None):
        # MPC 和安全层只读取已接入会话；尾部保证这些已知任务及 DR 日末约束可达。
        sessions=[s for s in self.sessions if s['departure_step']>self.t and (oracle or s['arrival_step']<=self.t)]
        h=self.config.horizon-self.t
        rows=[self.row(t) for t in range(self.t,self.config.horizon)] if oracle else [(fixed_row or self.row()).copy() for _ in range(h)]
        return solve_flex(self.spec,self.flex,self.network,rows,self.t,self.soc[0],self.backlog,self.shifted,self.shed_used,
            sessions,self.remaining,self.config.dt_hours,self.config.terminal_soc_penalty,self.config.solver_time_limit,
            proposal=proposal,objective=objective,objective_steps=h if oracle else min(lookahead,h))

    def _allocate(self,total):
        result={s['id']:0.0 for s in self.sessions};left=max(0,total)
        for s in sorted(self.active(),key=lambda s:(s['departure_step'],s['id'])):
            q=min(left,s['max_kw'],max(0,self.remaining[s['id']])/self.config.dt_hours)
            result[s['id']]=q;left-=q
        return result

    def check(self,row,a,allocation):
        dt=self.config.dt_hours;s=self.spec;f=self.flex;e,ev,shift,shed,pv,wind=a
        next_soc=s.next_soc(self.soc,[e,0],dt)[0]
        backlog=self.backlog+shift*dt
        flags=dict(ess_power=abs(e)>s.power_max[0]+1e-5,ess_soc=not s.soc_min[0]-1e-7<=next_soc<=s.soc_max[0]+1e-7,
            ev_station=not -1e-5<=ev<=f.ev_station_kw+1e-5,ev_sum=abs(sum(allocation.values())-ev)>1e-5,
            shift_power=not -f.dr_repay_kw-1e-5<=shift<=f.dr_shift_kw+1e-5,
            shed_power=not -1e-5<=shed<=f.dr_shed_kw+1e-5,dr_fraction=max(0,shift)+shed>f.dr_fraction*row['load_kw']+1e-5,
            dr_backlog=not -1e-5<=backlog<=f.dr_backlog_kwh+1e-5,
            dr_shift_budget=self.shifted+max(shift,0)*dt>f.dr_shift_budget_kwh+1e-5,
            dr_shed_budget=self.shed_used+shed*dt>f.dr_shed_budget_kwh+1e-5,
            dr_reachable=backlog>f.dr_repay_kw*dt*(self.config.horizon-self.t-1)+1e-5,
            pv_curtail=not -1e-5<=pv<=row['pv_kw']+1e-5,wind_curtail=not -1e-5<=wind<=row['wind_kw']+1e-5)
        for session in self.sessions:
            key=session['id'];q=allocation.get(key,0.);active=session['arrival_step']<=self.t<session['departure_step']
            after=self.remaining[key]-q*dt
            flags['ev_'+key]=q<-1e-5 or q>(session['max_kw'] if active else 0)+1e-5 or after<-1e-5
            if active:flags['ev_deadline_'+key]=after>session['max_kw']*max(0,session['departure_step']-self.t-1)*dt+1e-5
        grid=grid_power(row,a);flags['grid']=not -s.grid_export-1e-5<=grid<=s.grid_import+1e-5
        return int(sum(flags.values()))+self.network.linear(row,a)['linear_network_violations']

    def step(self,actions):
        raw=np.asarray(actions)
        if raw.shape!=(6,1) or not np.isfinite(raw).all():raise ValueError('会话模型动作应为有限 (6,1) 数组')
        x=np.tanh(raw[:,0]);r=self.row()
        shift=x[2]*(self.flex.dr_shift_kw if x[2]>=0 else self.flex.dr_repay_kw)
        a=np.array([x[0]*self.spec.power_max[0],(x[1]+1)*self.flex.ev_station_kw/2,shift,
                    (x[3]+1)*self.flex.dr_shed_kw/2,(x[4]+1)*r['pv_kw']/2,(x[5]+1)*r['wind_kw']/2])
        return self.step_physical(a)

    def step_physical(self,control):
        if self.t>=self.config.horizon:raise RuntimeError('回合已结束')
        planned=isinstance(control,dict)
        a=np.asarray(control['action'] if planned else control,dtype=float)
        if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('物理动作必须包含六个有限数')
        allocation=dict(control['ev_kw']) if planned else self._allocate(a[1])
        if not set(allocation)<=set(self.remaining) or not all(np.isfinite(v) for v in allocation.values()):raise ValueError('未知车辆或无效分车功率')
        row=self.row();pre=self.check(row,a,allocation);requested=a.copy();solver_seconds=0.
        if self.config.safety and (not planned or pre):
            plan,meta=self.plan(proposal=a);a=plan[0]['action'];allocation=plan[0]['ev_kw'];solver_seconds=meta['solver_seconds']
        count=self.check(row,a,allocation);dt=self.config.dt_hours
        departures=[s for s in self.active() if s['departure_step']==self.t+1]
        for key,q in allocation.items():self.remaining[key]-=q*dt
        unmet=sum(max(0,self.remaining[s['id']]) for s in departures)
        self.soc[0]=self.spec.next_soc(self.soc,[a[0],0],dt)[0]
        self.backlog+=a[2]*dt;self.shifted+=max(a[2],0)*dt;self.shed_used+=a[3]*dt
        grid=grid_power(row,a);cost=self.flex.cost(self.spec,grid,a,row['price'],dt)
        self.t+=1;done=self.t==self.config.horizon
        terminal=self.config.terminal_soc_penalty*max(0,self.spec.target_soc[0]-self.soc[0]) if done else 0.
        audit=self.network.audit(row,a)
        reward=-(cost+terminal+self.config.violation_penalty*count)/self.config.reward_scale
        info=dict(cost=cost,reward=float(reward),constraint_violations=count,pre_shield_violations=pre,shield_l1_kw=float(abs(a-requested).sum()),
            shield_solver_seconds=solver_seconds,ess_soc=float(self.soc[0]),ess_power_kw=float(a[0]),ev_charge_kw=float(a[1]),
            ev_departures=len(departures),ev_unmet_kwh=float(unmet),ev_delivered_kwh=float(a[1]*dt),
            ev_allocation_json=__import__('json').dumps(allocation,sort_keys=True),dr_shift_kw=float(a[2]),dr_shed_kw=float(a[3]),
            dr_backlog_kwh=float(self.backlog),dr_shifted_kwh=float(max(a[2],0)*dt),dr_repaid_kwh=float(max(-a[2],0)*dt),dr_shed_kwh=float(a[3]*dt),
            pv_curtail_kw=float(a[4]),wind_curtail_kw=float(a[5]),curtailment_kwh=float((a[4]+a[5])*dt),
            pv_used_kw=float(row['pv_kw']-a[4]),wind_used_kw=float(row['wind_kw']-a[5]),grid_power_kw=grid,terminal_penalty=float(terminal),
            **self.network.linear(row,a),**audit)
        info['ac_cost']=self.flex.cost(self.spec,audit['ac_grid_kw'],a,row['price'],dt) if audit['ac_converged'] else None
        if self.config.metrics_enabled:self.add_metrics(info,row)
        obs,share=self.encode()
        return obs,share,np.full((6,1),info['reward'],np.float32),done,info

    def add_metrics(self,info,row):
        from .objectives import carbon_kg,OBJECTIVE_NAMES
        valid=info['constraint_violations']==0;up=down=sym=seconds=0.;low=high=None;baseline_ok=False
        if valid and self.t<self.config.horizon:
            values=[]
            for mode in ('min_grid','max_grid'):
                plan,meta=self.plan(objective=mode,fixed_row=row)
                if not meta['solver_optimal']:raise RuntimeError('备用求解未达最优')
                values.append(grid_power(row,plan[0]['action']));seconds+=meta['solver_seconds']
            low,high=values;g=info['grid_power_kw'];up=max(0,g-low);down=max(0,high-g);baseline_ok=low-1e-5<=g<=high+1e-5
            sym=min(up,down) if baseline_ok else 0.
        kg=carbon_kg(info['grid_power_kw'],self.config.dt_hours,row['carbon_g_per_kwh'])
        flex=sym*self.config.dt_hours
        vector=np.array([-(info['cost']+info['terminal_penalty']),-kg,flex])/self.config.objective_scales
        info.update(dict(zip(OBJECTIVE_NAMES,map(float,vector))))
        info.update(carbon_g_per_kwh=row['carbon_g_per_kwh'],carbon_kg=kg,ac_carbon_kg=carbon_kg(info['ac_grid_kw'],self.config.dt_hours,row['carbon_g_per_kwh']) if info['ac_converged'] else None,
            flexibility_kw_hours=flex,reserve_up_kw=up,reserve_down_kw=down,reserve_symmetric_kw=sym,reserve_baseline_feasible=baseline_ok,
            reserve_min_grid_kw=low,reserve_max_grid_kw=high,reserve_solver_seconds=seconds,reserve_valid=valid)
        if self.config.algorithm=='weighted_mappo':info['reward']=float(np.dot(self.config.objective_weights,vector)-self.config.violation_penalty*info['constraint_violations']/self.config.reward_scale)


def resource_totals(rows):
    keys=('ev_departures','ev_unmet_kwh','ev_delivered_kwh','dr_shifted_kwh','dr_repaid_kwh','dr_shed_kwh','curtailment_kwh','shield_solver_seconds')
    result={k:sum(r[k] for r in rows) for k in keys}
    result['dr_terminal_backlog_kwh']=rows[-1]['dr_backlog_kwh']
    return result
