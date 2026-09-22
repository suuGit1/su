"""逐步记录公开观察、候选命令、协调结果和物理执行，便于闭环审计。"""
import json
from pathlib import Path
import numpy as np


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def append(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,default=plain,allow_nan=False)+'\n')


def step_record(env,obs,raw,preference,info,**tags):
    coordinator=json.loads(info['coordinator_json'])
    if coordinator is not None:
        coordinator['requested_preference']=list(preference)
    return dict(**tags,agent_names=env.agent_names,observation=obs.tolist(),
        dt_estimate=obs[0,:18].tolist(),dt_halfwidth=obs[0,54:63].tolist(),
        dt_ood=bool(obs[0,63]),preference=list(preference),
        raw_policy_action=None if raw is None else np.asarray(raw).tolist(),
        energy_candidate=json.loads(info['requested_energy_action_json']),
        coordinator=coordinator,bandwidth_bps=json.loads(info['bandwidth_allocated_bps_json']),
        cpu_cycles_per_second=json.loads(info['cpu_allocated_cycles_per_second_json']),
        executed_energy=[info[k] for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')],
        feedback=info)
