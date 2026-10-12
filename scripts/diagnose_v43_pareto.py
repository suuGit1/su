"""从已保存轨迹定位偏好响应、优化和执行差异；不重新训练或改动测试集。"""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vpp_research.pareto_v4 import ParetoV4
from vpp_research.pareto_agent import vector_gae
root=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/v43_pilot')
folder=root/'pareto_43'
saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
agent=ParetoV4(80,saved['n_agents'],saved['config']['hidden_size']);agent.load_state_dict(saved['agent']);agent.eval()
traces=[json.loads(s) for s in (folder/'trajectory_complete.jsonl').read_text().splitlines()]
obs=np.asarray([r['observation'] for r in traces],np.float32)
w=np.asarray([r['preference'] for r in traces],np.float32)
actions=np.asarray([r['raw_policy_action'] for r in traces],np.float32)
reward=np.asarray([r['feedback']['objective_vector'] for r in traces],np.float32)
preferences=np.r_[np.eye(3),[[1/3]*3]]
with torch.no_grad():
    means=np.asarray([agent.distribution(torch.tensor(obs),torch.tensor(p,dtype=torch.float32)).mean.numpy() for p in preferences])
span=np.ptp(means,axis=0)
history=list(csv.DictReader((folder/'training.csv').open()))
result={'scope':'既有单次更新验收轨迹诊断；不能外推收敛表现',
 'episodes':len(history),'updates':len({r['update_env_steps'] for r in history}),
 'training_preferences':[json.loads(r['preference']) for r in history],
 'actor_same_state_preference_span':{'mean':float(span.mean()),'max':float(span.max()),'energy_mean':float(span[:,:6].mean()),'comm_mean':float(span[:,6:12].mean()),'cpu_mean':float(span[:,12:].mean()),'exploration_std':agent.log_std.exp().tolist()},
 'reward_std':reward.std(0).tolist(),'reward_correlation':np.corrcoef(reward.T).tolist(),
 'training_budget':json.loads((folder/'budget_ledger.json').read_text()),
 'training_value_mse':[float(history[-1]['value_mse_'+k]) for k in ('cost','carbon','reserve')]}
# 此验收只有一次更新，可从相同随机种子准确重建 rollout 前的 critic。
if result['updates']==1:
    torch.manual_seed(saved['config']['seed'])
    initial=ParetoV4(80,saved['n_agents'],saved['config']['hidden_size'])
    with torch.no_grad():values=initial.value(torch.tensor(obs),torch.tensor(w)).numpy()
    dones=np.array([(i+1)%24==0 for i in range(len(obs))])
    adv,_=vector_gae(reward,np.r_[values,np.zeros((1,3))],dones,saved['config']['gamma'],saved['config']['gae_lambda'])
    scalar=(adv*w).sum(1)
    groups=[];grads=[]
    for ep in range(len(history)):
        sl=slice(ep*24,(ep+1)*24)
        # 逐偏好梯度只用于检查方向，不修改真实更新。
        dist=initial.distribution(torch.tensor(obs[sl]),torch.tensor(w[sl,None,:]))
        objective=(dist.log_prob(torch.tensor(actions[sl])).squeeze(-1)*torch.tensor(scalar[sl,None])).mean()
        grad=torch.autograd.grad(objective,list(initial.actor.parameters()))
        grads.append(torch.cat([g.flatten() for g in grad]).numpy())
        groups.append({'episode':ep+1,'scalar_adv_mean':float(scalar[sl].mean()),'scalar_adv_std':float(scalar[sl].std()),'positive_fraction_after_global_center':float((scalar[sl]>scalar.mean()).mean())})
    g=np.asarray(grads);norm=np.linalg.norm(g,axis=1)
    result['first_update']={'adv_std':adv.std(0).tolist(),'weighted_component_std':(adv*w).std(0).tolist(),'per_episode':groups,'episode_gradient_cosine':(g@g.T/(norm[:,None]*norm[None,:]+1e-20)).tolist(),'gradient_scope':'不同日期和偏好同时变化；夹角不能单独证明目标冲突'}
rows=[]
for p in sorted((folder/'test').glob('weight_*/*/trajectory_complete.jsonl')):
    rr=[json.loads(s) for s in p.read_text().splitlines()]
    vec=np.sum([r['feedback']['objective_vector'] for r in rr],axis=0)
    issued=np.array([r['issued_energy_candidate'] for r in rr]);received=np.array([r['received_energy_candidate'] for r in rr]);executed=np.array([r['executed_energy'] for r in rr])
    rows.append({'path':str(p),'preference':rr[0]['preference'],'vector':vec.tolist(),'received_executed_l1_kw_sum':float(np.abs(received-executed).sum()),'issued_abs_kw_sum':float(np.abs(issued).sum()),'terminal_penalty':sum(r['feedback']['terminal_penalty'] for r in rr)})
result['test_execution']=rows
out=Path('experiments/v43/pareto_diagnosis.json');out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='training_budget'},ensure_ascii=False,indent=2))
