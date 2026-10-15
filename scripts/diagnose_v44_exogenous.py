"""区分外生负荷背景与可控奖励波动；不修改训练奖励或物理目标。"""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vpp_mappo.data import CSVProfiles
from vpp_research.pareto_agent import vector_gae
root=Path('runs/v44_value_repair/conditioned')
meta=json.loads((root/'metadata.json').read_text());cfg=meta['config'];data=CSVProfiles(cfg['train_csv'],cfg['horizon'])
history=list(csv.DictReader((root/'training.csv').open()));days={i:int(r['training_day']) for i,r in enumerate(history)}
rows=[json.loads(s) for s in (root/'trajectory_complete.jsonl').read_text().splitlines()]
original=[];baselines=[];dones=[];prefs=[]
for r in rows:
 p=data.profiles[days[r['episode']]];t=r['step'];net=p['load_kw'][t]-p['pv_kw'][t]-p['wind_kw'][t]
 cost=cfg['dt_hours']*p['price'][t]*(max(net,0)-meta['dispatch_spec']['sell_ratio']*max(-net,0))
 carbon=cfg['dt_hours']*max(net,0)*p['carbon_g_per_kwh'][t]/1000
 original.append(r['feedback']['objective_vector']);baselines.append(np.array([-cost,-carbon,0])/100);dones.append(t==cfg['horizon']-1);prefs.append(r['preference'])
original=np.asarray(original);baselines=np.asarray(baselines);centered=original-baselines
result=dict(scope='仅诊断固定时域、外生轨迹独立于控制的背景信号；不证明控制提升，未用于模型训练',reward_std_original=original.std(0).tolist(),reward_std_exogenous_centered=centered.std(0).tolist())
for label,r in [('original',original),('exogenous_centered',centered)]:
 adv,_=vector_gae(r,np.zeros((len(r)+1,3)),dones,cfg['gamma'],cfg['gae_lambda'])
 result[label+'_zero_baseline_gae_std']=adv.std(0).tolist()
result['identity_max_error']=float(abs(centered+baselines-original).max())
Path('experiments/v44/exogenous_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
