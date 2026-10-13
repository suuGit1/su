"""汇总同学习器端点诊断；分别呈现控制代价、终端储能和偏好效用。"""
import json,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.suite import feasible
root=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/v44_diagnosis')
frozen=json.loads((root/'frozen.json').read_text());cases=[]
for name in frozen['arguments']['cases']:
 folder=root/name
 if not (folder/'result.json').exists():raise RuntimeError(name+'未完成，禁止生成完成报告')
 result=json.loads((folder/'result.json').read_text());history=list(csv.DictReader((folder/'training.csv').open()))
 metrics=[]
 for path in sorted((folder/'diagnostic_evaluation').glob('*/trajectory_complete.jsonl')):
  rows=[json.loads(s) for s in path.read_text().splitlines()];first=rows[0]
  obs=np.asarray([r['observation'] for r in rows]);issued=np.asarray([r['issued_energy_candidate'] for r in rows]);executed=np.asarray([r['executed_energy'] for r in rows]);received=np.asarray([r['received_energy_candidate'] for r in rows])
  metrics.append(dict(preference=first['preference'],cost=sum(r['feedback']['ac_cost'] for r in rows),terminal_penalty=sum(r['feedback']['terminal_penalty'] for r in rows),terminal_soc=rows[-1]['feedback']['ess_soc'],carbon_kg=sum(r['feedback']['carbon_kg'] for r in rows),reserve_kwh=sum(r['feedback']['flexibility_kwh'] for r in rows),inactive_age_max=float(abs(obs[:,:, [19,21]]).max()),inactive_ev_candidate_max=float(abs(issued[:,1]).max()),energy_correction_by_channel=np.abs(received-executed).sum(0).tolist(),last_two_steps=[dict(step=r['step'],issued_ess=r['issued_energy_candidate'][0],received_ess=r['received_energy_candidate'][0],executed_ess=r['executed_energy'][0],soc=r['feedback']['ess_soc']) for r in rows[-2:]]))
 last=history[-1]
 cases.append(dict(case=name,train_steps=result['training']['attempted_calls'],train_unknown=result['training']['unknown_calls'],train_errors=result['training']['error_calls'],episodes=len(history),updates=len({r['update_env_steps'] for r in history}),training_days=[int(r['training_day']) for r in history],preference_action_span=float(last['preference_action_span']),value_explained_variance={k:float(last['value_explained_variance_'+k]) for k in ('cost','carbon','reserve')},evaluation=result['results'],physical_breakdown=metrics))
comparisons=[]
conditioned=next(c for c in cases if c['case']=='conditioned')
for row in conditioned['evaluation']:
 w=np.asarray(row['preference']);alternatives=[]
 for c in cases:
  if c['case']=='conditioned':continue
  match=[r for r in c['evaluation'] if r['seed']==row['seed'] and np.allclose(r['preference'],w)]
  if len(match)!=1:raise RuntimeError('配对开发场景不完整')
  r=match[0]
  if feasible(r):alternatives.append((float(w@r['vector']),c['case']))
 if alternatives and feasible(row):
  best=max(alternatives);utility=float(w@row['vector']);comparisons.append(dict(preference=w.tolist(),seed=row['seed'],conditioned_utility=utility,best_endpoint_utility=best[0],best_endpoint=best[1],empirical_utility_gap=best[0]-utility))
report=dict(scope='同实现机制诊断；单策略训练交互相等，端点家族合计预算取决于参与案例数；无选模、无确认性测试，不是收敛优越性证据',cases=cases,paired_utility=comparisons,matched_train_steps=len({c['train_steps'] for c in cases})==1,matched_training_days=all(c['training_days']==cases[0]['training_days'] for c in cases))
out=Path(sys.argv[2] if len(sys.argv)>2 else 'experiments/v44/diagnostic_report.json');out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))
for c in cases:print(c['case'],c['train_steps'],c['value_explained_variance'],c['physical_breakdown'][0])
