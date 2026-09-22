"""固定 65 维 C3 观察及经济—碳—备用协议；旧版环境继续独立保留。"""
import copy
from dataclasses import replace
import numpy as np
from gymnasium.spaces import Box
from vpp_mappo.cyber_environment import CyberVPPAdapter
from vpp_mappo.flex_resources import FlexNetwork
from vpp_mappo.objectives import carbon_kg
from .dt import TARGETS,predict,VERSION as DT_VERSION
from .safety import guard

OBS_VERSION='c3-65-state-interval-carbon-v1'
OBJECTIVE_VERSION='c3-cost-carbon-symmetric-reserve-v1'
AC_OBJECTIVE_VERSION='c3-cost-carbon-ac-endpoint-reserve-v2'
SCALES=np.array([100.,100.,100.])


class ResearchEnv(CyberVPPAdapter):
    def __init__(self,config,csv_path=None,dt_model=None,robust=False,vector_metrics=True,ac_safe=False,resource_mode="joint",timing_contract=None,control_cycles=0.,reserve_mode="linear"):
        if reserve_mode not in ("linear","ac_checked"):raise ValueError("未知备用目标版本")
        self.reserve_mode=reserve_mode
        self.objective_version=AC_OBJECTIVE_VERSION if reserve_mode=="ac_checked" else OBJECTIVE_VERSION
        if resource_mode not in ("control", "communication", "computation", "joint"):
            raise ValueError("未知 C3 消融模式")
        self.resource_mode=resource_mode
        self.timing_contract=timing_contract;self.control_cycles=float(control_cycles)
        if not np.isfinite(self.control_cycles) or self.control_cycles<0:raise ValueError("控制计算周期必须有限非负")
        if config.reward_scale!=100.:raise ValueError('研究协议固定经济尺度为 100，不能改变 reward_scale')
        self.dt_model=copy.deepcopy(dt_model);self.robust=robust;self.vector_metrics=vector_metrics
        if robust and dt_model is None:raise ValueError('区间保护要求独立校准的 DT 模型')
        if dt_model is not None:
            widths=np.asarray(dt_model.get('halfwidth',[]))
            if dt_model.get('version')!=DT_VERSION or widths.shape!=(9,) or not np.isfinite(widths).all() or np.any(widths<0):
                raise ValueError('DT 校准模型版本或区间非法')
            config=copy.copy(config);config.cyber_dt_mode='hold' if dt_model['method']=='hold' else 'physics'
        super().__init__(config,csv_path)
        if ac_safe:
            from .ac_safety import ACCheckedFlex
            self.core=ACCheckedFlex(config,csv_path)
        self.network=FlexNetwork(self.spec)
        if vector_metrics:
            if self.data and any('carbon_g_per_kwh' not in p for p in self.data.profiles):raise ValueError('真实三目标实验缺少碳数据；禁止静默填补')
            if not self.data and config.synthetic_carbon_g_per_kwh is None:raise ValueError('合成碳假设必须显式指定')
        self.observation_space=Box(-np.inf,np.inf,(65,),dtype=np.float32)
        self.share_observation_space=Box(-np.inf,np.inf,(65*self.num_agents,),dtype=np.float32)
        self.preference=np.array([1.,0.,0.]);self.reward_mode='economic'

    def reset(self,scenario_seed,profile_index=0):
        from .timing import CommandQueue
        self.command_queue=CommandQueue(self.timing_contract) if self.timing_contract else None
        return super().reset(scenario_seed,profile_index)

    def sensor_payloads(self):
        data=super().sensor_payloads();r=self.core.row()
        data[2]['carbon_g_per_kwh']=r.get('carbon_observed_g_per_kwh',r.get('carbon_g_per_kwh',self.config.synthetic_carbon_g_per_kwh or 0.))
        return data

    def raw_observation(self):return super().encode()[0][0].copy()

    def encode(self):
        if self.t>=self.config.horizon:
            return np.zeros((self.num_agents,65),np.float32),np.zeros((self.num_agents,65*self.num_agents),np.float32)
        obs,_=super().encode();width=np.zeros(9);ood=False
        if self.dt_model:
            prediction,ood=predict(self.dt_model,obs[0]);obs[:,TARGETS]=prediction
            width=np.asarray(self.dt_model['halfwidth'])
            obs[:,13]=(obs[:,9]*5000-obs[:,11]*1000-obs[:,12]*1000)/5000
        carbon=self.dt.cache[2].get('carbon_g_per_kwh',0)/1000
        ext=np.tile(np.r_[width,float(ood),carbon],(self.num_agents,1));obs=np.c_[obs,ext].astype(np.float32)
        return obs,np.repeat(obs.reshape(1,-1),self.num_agents,axis=0)

    def step(self,actions):
        raw=np.asarray(actions,dtype=float)
        if raw.shape!=(self.num_agents,1) or not np.isfinite(raw).all():raise ValueError('研究动作维度或数值错误')
        obs=self.encode()[0][0];x=np.tanh(raw[:,0])
        a=np.array([x[0]*self.spec.power_max[0],(x[1]+1)*self.flex.ev_station_kw/2,
            x[2]*(self.flex.dr_shift_kw if x[2]>=0 else self.flex.dr_repay_kw),(x[3]+1)*self.flex.dr_shed_kw/2,
            (x[4]+1)*max(0,obs[11]*1000)/2,(x[5]+1)*max(0,obs[12]*1000)/2])
        bw=np.maximum(0,x[6:12]) if self.num_agents==18 else np.ones(6)
        cpu=np.maximum(0,x[12:18]) if self.num_agents==18 else np.ones(6)
        if self.resource_mode in ("control", "computation"):bw=np.ones(6)
        if self.resource_mode in ("control", "communication"):cpu=np.ones(6)
        return self.step_candidate(a,bw,cpu)

    def step_candidate(self,energy,bw,cpu):
        timing_meta={};control_energy=0.
        if self.command_queue is not None:
            capacity=self.cyber_spec.cpu_cycles_per_second;dt_seconds=self.config.dt_hours*3600
            needed=self.control_cycles/capacity if capacity>0 else float('inf')
            if np.isfinite(needed):delivery=self.command_queue.submit(self.pipeline.now,energy,needed,0.)
            else:delivery=dict(delivered=False,seconds=None,deadline_missed=True)
            received,expired=self.command_queue.receive(self.pipeline.now)
            if received is None:
                # 下行缺失时由同一现场约束的经济规划提供后备；不可行仍显式失败。
                backup,_=self.core.plan();energy=backup[0]['action']
            else:energy=received
            used=min(self.control_cycles,capacity*dt_seconds)
            available=max(0.,capacity-used/dt_seconds)
            self.pipeline.spec=replace(self.cyber_spec,cpu_cycles_per_second=available)
            control_energy=used*self.cyber_spec.joules_per_cycle
            timing_meta=dict(command_latency_seconds=delivery['seconds'],command_deadline_missed=delivery['deadline_missed'],
                command_expired=expired,local_fallback=received is None,control_cycles=used,dt_available_cpu_cycles_per_second=available,
                command_queue_length=len(self.command_queue.pending),timing_provenance=self.timing_contract.provenance)
        public=self.encode()[0][0];truth=self.features(self.sensor_payloads())[TARGETS]
        squared=float(np.mean((public[TARGETS]-truth)**2));covered=bool(np.all(np.abs(public[TARGETS]-truth)<=public[54:63]+1e-7))
        row=self.core.row().copy();certificate=None;meta=dict(guard_feasible=False,guard_status='disabled',guard_seconds=0.)
        if self.robust:
            energy,meta,certificate=guard(energy,self.encode()[0][0],self.spec,self.flex,self.network,self.config.dt_hours,self.config.horizon)
        obs,share,rewards,done,info=super().step_candidate(energy,bw,cpu)
        info.update(timing_meta)
        if control_energy:
            extra=control_energy/3.6e6*self.cyber_spec.energy_price_per_kwh
            info['cyber_energy_j']+=control_energy;info['cyber_cost']+=extra;info['cost']+=extra
            info['reward']-=extra/self.config.reward_scale;rewards-=extra/self.config.reward_scale
            if info['ac_cost'] is not None:info['ac_cost']+=extra
        executed=np.array([info[k] for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')])
        info.update(research_dt_mse=squared,interval_covered=covered,**meta,guard_certificate_survived=certificate(executed) if certificate else False)
        if self.vector_metrics:
            # 仅事后奖励核算使用物理执行结果；策略碳观察仍通过遥测。
            factor=row.get('carbon_g_per_kwh',self.config.synthetic_carbon_g_per_kwh)
            cyber_kg=info['cyber_energy_j']/3.6e6*factor/1000
            kg=carbon_kg(info['grid_power_kw'],self.config.dt_hours,factor)+cyber_kg
            reserve=0.;reserve_valid=info['constraint_violations']==0
            if reserve_valid and not done:
                values=[]
                try:
                    for mode in ('min_grid','max_grid'):
                        plans,m=self.core.plan(objective=mode,fixed_row=row)
                        if not m['solver_optimal']:raise RuntimeError('备用包络未达最优')
                        from vpp_mappo.flex_resources import grid_power
                        values.append(grid_power(row,plans[0]['action']))
                    g=info['grid_power_kw'];lo,hi=values
                    reserve=max(0,min(g-lo,hi-g))*self.config.dt_hours if lo-1e-5<=g<=hi+1e-5 else 0.
                except (RuntimeError,ValueError):reserve_valid=False
            if self.reserve_mode=='ac_checked' and reserve_valid and not done:
                from .checked_reserve import checked_capacity
                try:
                    confirmation=checked_capacity(self.core,info['grid_power_kw'],reserve/self.config.dt_hours)
                    reserve=confirmation['kw']*self.config.dt_hours
                    info['reserve_confirmation']=confirmation
                except (RuntimeError,ValueError) as exc:
                    reserve=0.;reserve_valid=False;info['reserve_confirmation_error']=str(exc)
            vector=np.array([-(info['cost']+info['terminal_penalty']),-kg,reserve])/SCALES
            penalty=self.config.violation_penalty*info['constraint_violations']/self.config.reward_scale
            reward=float(vector[0] if self.reward_mode=='economic' else self.preference@vector)-penalty
            info.update(objective_version=self.objective_version,objective_vector=vector.tolist(),carbon_kg=kg,
                ac_carbon_kg=carbon_kg(info['ac_grid_kw'],self.config.dt_hours,factor)+cyber_kg if info['ac_converged'] else None,
                flexibility_kwh=reserve,reserve_valid=reserve_valid,reward=reward)
            rewards=np.full((self.num_agents,1),reward,np.float32)
        return obs,share,rewards,done,info
