"""第一二阶段机制研究；开发日期与封存日期隔离，固定协议后运行。"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.cyber import CyberSpec
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_mappo.optimization import DispatchInfeasible
from vpp_research.dt import predict,interval_halfwidth,TARGETS,assess
from vpp_research.environment import ResearchEnv
from vpp_research.real_inputs import connect
from vpp_mappo.coordinator_service import ServiceCoordinator


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)


def case(task):
    c,csv,model,design,out=task;file=out/'cases'/(design['id']+'.json')
    if file.exists():return json.loads(file.read_text())
    c=copy.deepcopy(c);c.coordinator_mode='off';c.risk_adaptive=False
    variant=design.get('variant','uniform');c.task_focus=design.get('task','balanced')
    if variant in ('joint_schedule','joint_risk'):c.coordinator_mode='schedule';c.risk_adaptive=variant=='joint_risk'
    c._cyber_record={**asdict(CyberSpec.load(c.cyber_spec)),**design.get('cyber',{})}
    fault=design.get('fault');c.interval_safety=fault!='no_interval'
    if fault=='downlink_loss':c.downlink_bps=0.
    result=dict(design=design,failed=False,steps=[])
    env=None
    try:
        env=ResearchEnv(c,csv,dt_model=model,reserve_mode='pcc_checked');env.coordinator=ServiceCoordinator(c,env.spec,env.flex,[1,0,1,0,1,1],value_priority=design['revision']=='service_value');obs,_=env.reset(design['seed'],design['index'])
        if fault=='load_drift':env.core.profiles['load_kw'][12:]*=1.2
        if fault=='resource_shortage':
            # 故障注入：变更并网能力，无法满足硬约束时必须拒绝推进。
            env.core.spec=copy.deepcopy(env.core.spec)
            env.core.spec.grid_import=0.;env.core.spec.grid_export=0.
        planner=ObservedMPC(c,env.spec,env.flex,2)
        for t in range(c.horizon):
            before=obs[0].copy();action,meta=planner.propose(before)
            age=1+np.maximum(0,before[18:24]);cpu=1+np.maximum(0,before[30:36])+np.maximum(0,before[18:24])
            active=np.array([1.,0.,1.,0.,1.,1.]);bw=np.ones(6) if variant=='uniform' else active.copy();cw=bw.copy()
            if variant in ('communication_age','joint_age','joint_schedule','joint_risk'):bw=active*age;bw/=bw.sum()
            if variant in ('computation_age','joint_age','joint_schedule','joint_risk'):cw=active*cpu;cw/=cw.sum()
            inject=(patch.object(env.core,'plan',side_effect=DispatchInfeasible('预注册求解超时故障',reason='timeout')) if fault=='solver_timeout' and t==12
                    else patch.object(env.core.network,'audit',return_value=dict(ac_converged=False,ac_violations=1)) if fault=='ac_rejection' and t==12 else nullcontext())
            tx,cp=env.pipeline.queue_state();seconds=c.dt_hours*3600
            env.coordinator.public_state=before.copy()
            env.coordinator.service_state=dict(tx_bits=tx,cpu_cycles=cp,tx_count=[len(q) for q in env.pipeline.tx],
                cpu_count=[len(q) for q in env.pipeline.cpu],queue_capacity=env.cyber_spec.queue_capacity,
                deadline_seconds=seconds,bandwidth_bps=env.cyber_spec.bandwidth_bps,
                cpu_cycles_per_second=max(0.,env.cyber_spec.cpu_cycles_per_second-min(c.control_cycles,env.cyber_spec.cpu_cycles_per_second*seconds)/seconds),
                packet_overhead_bits=env.cyber_spec.packet_overhead_bits,payload_bound_bits=8192.,update_cycles=env.cyber_spec.update_cycles)
            with inject:obs,_,_,done,i=env.step_candidate(action,bw,cw)
            result['steps'].append(dict(step=t,cost=float(i['objective_cost']+i['terminal_penalty']),carbon=float(i['carbon_kg']),reserve=float(i['flexibility_kwh']),
                mse=float(i['research_dt_mse']),covered=bool(i['interval_covered']),certificate=bool(i['guard_certificate_survived']),
                guard_reason=i.get('guard_reason'),ac_violations=int(i['ac_violations']),reserve_valid=bool(i['reserve_valid']),
                aoi=float(i['aoi_mean_seconds']),ood=bool(before[63]),width=before[54:63].tolist(),
                bw=json.loads(i['bandwidth_allocated_bps_json']),cpu=json.loads(i['cpu_allocated_cycles_per_second_json']),
                intervention=float(i['shield_l1_kw']),pre_violations=int(i['pre_shield_violations']),fallback=bool(i.get('local_fallback',False)),
                safety_mode=i.get('safety_mode'),safety_hard_verified=bool(i.get('safety_hard_verified',False)),
                cyber_events=json.loads(i['cyber_events_json']),dt_deadline_misses=i['dt_deadline_misses'],
                aoi_by_channel_steps=before[18:24].tolist(),active_mean_aoi_steps=float(before[18:24][[0,2,4,5]].mean()),
                issued_action=i['issued_energy_candidate_kw'],executed_action=[float(i[k]) for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')],
                command_latency_seconds=i.get('command_latency_seconds'),safety_failures=i.get('safety_failures',[]),coordinator=json.loads(i['coordinator_json'])))
        result['completed']=bool(done)
    except (RuntimeError,ValueError) as exc:
        result.update(failed=True,error=str(exc),reason=getattr(exc,'reason',None),state_step=env.core.t if env else None)
    write(file,result);return result


def run(a):
    protocol=json.loads(Path(a.protocol).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    c,paths,sets,source=connect(a.data_root,Config.load('configs/v42_ieee33.json'))
    if any(not d.startswith('2019-03-') for d in sets['test'].scenario_names):raise ValueError('机制实验仅允许既有三月开发集，禁止读取确认集')
    if a.workers<1:raise ValueError('workers须为正')
    # 归因实验复用先前已经冻结的模型与离线标签，不根据机制结果重新调参。
    with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:
        base={k:json.loads(z.read(k+'.json')) for k in ('physics','residual')}
        labels=json.loads(z.read('test_records.json'));cal=json.loads(z.read('validation_records.json'))
    models={}
    for center in base:
        for width in base:
            m=copy.deepcopy(base[center]);m['halfwidth']=base[width]['halfwidth'];models[center+'_'+width]=m
    designs=[]
    def add(group,model,index,seed,**kwargs):
        designs.append(dict(id=f'{len(designs):05d}',group=group,model=model,index=index,date=sets['test'].scenario_names[index],seed=seed,**kwargs))
    for name,cyber in protocol['resource_settings'].items():
        for index in protocol['mechanism_day_indices']:
            for seed in (protocol['cyber_seeds'] if name=='combined' else protocol['cyber_seeds'][:1]):
                for revision in ('service_risk','service_value'):
                    add('c3','residual_residual',index,seed,condition=name,cyber=cyber,variant='joint_risk',revision=revision)
    for index in protocol['mechanism_day_indices']:
        for task in ('economic','carbon','reserve'):
            add('tasks','residual_residual',index,431,condition='combined',cyber=protocol['resource_settings']['combined'],variant='joint_risk',task=task,revision='service_value')
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in ('vpp_research','vpp_mappo') for p in sorted(Path(name).rglob('*.py'))}
    frozen=dict(protocol=protocol,source=source,config=c.__dict__,designs=designs,runner_sha256=fingerprint,source_hashes=sources,
                models_sha256=hashlib.sha256(json.dumps(models,sort_keys=True).encode()).hexdigest())
    frozen=json.loads(json.dumps(frozen))
    if (out/'frozen.json').exists() and json.loads((out/'frozen.json').read_text())!=frozen:raise ValueError('协议或实现已变更，不得混用；请另开版本')
    write(out/'frozen.json',frozen)
    offline={}
    for name,m in models.items():
        x=np.array([r['x'] for r in cal]);y=np.array([r['y'] for r in cal]);inside=abs(predict(m,x)[0]-y)<=interval_halfwidth(m,x)+1e-12
        covered=[bool(inside[[j for j,r in enumerate(cal) if r['scenario']==day]].all()) for day in sorted({r['scenario'] for r in cal})]
        offline[name]=dict(development=assess(m,labels),calibration_empirical_day_coverage=float(np.mean(covered)),
            scope='交换宽度的交叉消融只报告实测覆盖，不能继承原模型名义覆盖声明')
    write(out/'offline.json',offline)
    if a.freeze_only:print('已冻结',len(designs),'个完整日案例');return
    tasks=[(c,paths['test'],models[d['model']],d,out) for d in designs]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for n,r in enumerate(pool.map(case,tasks)):
            print(n+1,len(tasks),r['design']['group'],r['design']['id'],'failed',r['failed'],flush=True)
    print('全部案例完成；失败保留在各案例文件中',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='data/real');p.add_argument('--protocol',default='configs/phase12_protocol.json');p.add_argument('--output',default='runs/phase12_service');p.add_argument('--workers',type=int,default=6);p.add_argument('--freeze-only',action='store_true');run(p.parse_args())
