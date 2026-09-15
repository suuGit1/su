"""通信—计算—能源控制基础闭环；现场安全执行器与受限信息策略显式分离。"""
import json
import numpy as np
from gymnasium.spaces import Box
from .flex_environment import FlexVPPAdapter
from .cyber import CyberSpec,CyberPipeline,CHANNELS,CONTRACT_VERSION
from .basic_dt import BasicDT


class CyberVPPAdapter:
    action_space=Box(-np.inf,np.inf,(1,),dtype=np.float32)

    def __init__(self,config,csv_path=None):
        self.config=config;self.core=FlexVPPAdapter(config,csv_path)
        self.spec=self.core.spec;self.flex=self.core.flex;self.bundle=self.core.bundle;self.data=self.core.data
        self.cyber_spec=CyberSpec.load(getattr(config,'_cyber_record',None) or config.cyber_spec)
        self.agent_names=list(CHANNELS)
        if config.cyber_mode=='joint':self.agent_names += ['comm_'+k for k in CHANNELS]+['cpu_'+k for k in CHANNELS]
        self.num_agents=len(self.agent_names)
        # 固定布局：18 维 DT 状态、6 维 AoI、12 维队列、18 维角色标识。
        self.observation_space=Box(-np.inf,np.inf,(54,),dtype=np.float32)
        self.share_observation_space=Box(-np.inf,np.inf,(54*self.num_agents,),dtype=np.float32)

    def sensor_payloads(self):
        # 仅采样边界调用；信息不能直接进入策略输入。
        r=self.core.row()
        sessions=[dict(s,remaining_kwh=self.core.remaining[s['id']]) for s in self.core.active()]
        return [dict(soc=float(self.core.soc[0])),dict(sessions=sessions),
            dict(backlog=self.core.backlog,shifted=self.core.shifted,load_kw=r['load_kw'],price=r['price']),
            dict(shed_used=self.core.shed_used),dict(pv_kw=r['pv_kw']),dict(wind_kw=r['wind_kw'])]

    def reset(self,scenario_seed,profile_index=0):
        self.core.reset(scenario_seed,profile_index);self.t=0
        self.pipeline=CyberPipeline(self.cyber_spec,scenario_seed+9173)
        self.dt=BasicDT(self.spec,self.flex,self.config.dt_hours,self.sensor_payloads(),self.config.cyber_dt_mode)
        return self.encode()

    def features(self,data):
        ss=data[1]['sessions'];need=sum(s['remaining_kwh'] for s in ss)
        deadline=min((s['departure_step']-self.t for s in ss),default=0)
        return np.array([self.t/self.config.horizon,data[0]['soc'],self.spec.target_soc[0],need/1000,len(ss)/100,
            deadline/self.config.horizon,data[2]['backlog']/max(1,self.flex.dr_backlog_kwh),
            data[2]['shifted']/max(1,self.flex.dr_shift_budget_kwh),data[3]['shed_used']/max(1,self.flex.dr_shed_budget_kwh),
            data[2]['load_kw']/5000,data[2]['price']/.3,data[4]['pv_kw']/1000,data[5]['wind_kw']/1000,
            (data[2]['load_kw']-data[4]['pv_kw']-data[5]['wind_kw'])/5000,
            float(self.config.cyber_dt_mode=='physics'),self.cyber_spec.bandwidth_bps/10000,
            self.cyber_spec.cpu_cycles_per_second/1.e7,float(self.config.safety)],dtype=np.float32)

    def encode(self):
        if self.t>=self.config.horizon:
            return np.zeros((self.num_agents,54),np.float32),np.zeros((self.num_agents,54*self.num_agents),np.float32)
        data=self.dt.estimate(self.t)
        age=self.dt.age(self.pipeline.now)/(3600*self.config.dt_hours)
        tq,cq=self.pipeline.queue_state()
        queues=np.r_[np.asarray(tq)/max(1,self.cyber_spec.packet_overhead_bits),np.asarray(cq)/self.cyber_spec.update_cycles]
        common=np.r_[self.features(data),age,queues]
        obs=np.asarray([np.r_[common,np.eye(18)[i]] for i in range(self.num_agents)],np.float32)
        # CTDE 的价值网络也仅使用同一缓存信息，避免集中训练读取物理真值。
        return obs,np.repeat(obs.reshape(1,-1),self.num_agents,axis=0)

    def step(self,actions):
        raw=np.asarray(actions,dtype=float)
        if raw.shape!=(self.num_agents,1) or not np.isfinite(raw).all():raise ValueError('C3 动作维度或数值不合法')
        if self.t>=self.config.horizon:raise RuntimeError('回合已结束')
        x=np.tanh(raw[:,0]);data=self.dt.estimate(self.t)
        # 弃电动作按估计的可用出力解码，不借助物理真值缩放策略动作。
        energy=np.array([x[0]*self.spec.power_max[0],(x[1]+1)*self.flex.ev_station_kw/2,
            x[2]*(self.flex.dr_shift_kw if x[2]>=0 else self.flex.dr_repay_kw),(x[3]+1)*self.flex.dr_shed_kw/2,
            (x[4]+1)*data[4]['pv_kw']/2,(x[5]+1)*data[5]['wind_kw']/2])
        if self.config.cyber_mode=='joint':
            bw=np.maximum(0,x[6:12]);cpu=np.maximum(0,x[12:18])
        else:bw=cpu=np.ones(6)
        events=self.pipeline.enqueue(self.sensor_payloads(),bw>0)
        # 只把候选命令给 DT；安全层实际动作与分车结果没有零时延旁路。
        self.dt.command(energy)
        *_,done,info=self.core.step_physical(energy)
        completed,more,cyber=self.pipeline.advance(self.config.dt_hours*3600,bw,cpu);events+=more
        updated=sum(self.dt.receive(p) for p in completed)
        self.t+=1
        cost=cyber['cyber_energy_j']/3.6e6*self.cyber_spec.energy_price_per_kwh
        info['energy_dispatch_cost']=info['cost'];info['cyber_cost']=cost;info['cost']+=cost
        if info['ac_cost'] is not None:info['ac_cost']+=cost
        info['reward']-=cost/self.config.reward_scale
        ages=self.dt.age(self.pipeline.now);latencies=[p['finished']-p['sampled'] for p in completed]
        info.update(cyber_contract=CONTRACT_VERSION,dt_mode=self.config.cyber_dt_mode,dt_updates=updated,
            dt_completed_jobs=len(completed),aoi_mean_seconds=float(ages.mean()),aoi_max_seconds=float(ages.max()),
            dt_latency_sum_seconds=sum(latencies),dt_deadline_misses=sum(t>self.cyber_spec.deadline_seconds for t in latencies),
            packet_drops=sum(e['kind'] in ('tx_overflow','cpu_overflow','packet_loss') for e in events),
            tx_bits=cyber['tx_bits'],cpu_cycles=cyber['cpu_cycles'],cyber_energy_j=cyber['cyber_energy_j'],
            pending_overdue_jobs=sum(self.pipeline.now-p['sampled']>self.cyber_spec.deadline_seconds for q in self.pipeline.tx+self.pipeline.cpu for p in q),
            requested_energy_action_json=json.dumps(energy.tolist()),
            cyber_events_json=json.dumps(events,sort_keys=True),
            bandwidth_allocated_bps_json=json.dumps(cyber['bandwidth_allocated_bps']),
            cpu_allocated_cycles_per_second_json=json.dumps(cyber['cpu_allocated_cycles_per_second']),
            tx_queue_packets=sum(map(len,self.pipeline.tx)),cpu_queue_jobs=sum(map(len,self.pipeline.cpu)),
            safety_information='local_current_sensors')
        # 真值仅用于离线诊断，既不进入观察/价值函数，也不作为额外奖励。
        if not done:
            estimate=self.features(self.dt.estimate(self.t));truth=self.features(self.sensor_payloads())
            info['dt_state_rmse_normalized']=float(np.sqrt(np.mean((estimate[:14]-truth[:14])**2)))
        else:info['dt_state_rmse_normalized']=None
        obs,share=self.encode()
        return obs,share,np.full((self.num_agents,1),info['reward'],np.float32),done,info


def cyber_totals(rows):
    keys=('cyber_cost','cyber_energy_j','tx_bits','cpu_cycles','dt_updates','dt_completed_jobs','dt_latency_sum_seconds','dt_deadline_misses','packet_drops')
    total={k:sum(r[k] for r in rows) for k in keys}
    total['mean_aoi_seconds']=float(np.mean([r['aoi_mean_seconds'] for r in rows]))
    total['max_aoi_seconds']=max(r['aoi_max_seconds'] for r in rows)
    total['mean_completed_dt_latency_seconds']=total['dt_latency_sum_seconds']/total['dt_completed_jobs'] if total['dt_completed_jobs'] else None
    errors=[r['dt_state_rmse_normalized'] for r in rows if r['dt_state_rmse_normalized'] is not None]
    total['mean_dt_state_rmse_normalized']=float(np.mean(errors)) if errors else None
    total['safety_intervention_steps']=sum(r['shield_l1_kw']>1e-5 for r in rows)
    total['safety_correction_sum_kw']=sum(r['shield_l1_kw'] for r in rows)
    total['terminal_overdue_jobs']=rows[-1]['pending_overdue_jobs']
    total['terminal_tx_queue_packets']=rows[-1]['tx_queue_packets'];total['terminal_cpu_queue_jobs']=rows[-1]['cpu_queue_jobs']
    return total
