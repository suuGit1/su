"""独立检查critic能否拟合既有回报；不更新actor，不把拟合当作控制收益。"""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vpp_research.train import load
folder=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/v44_diagnosis/conditioned')
torch.set_num_threads(1)
state,c,agent=load(folder/'latest.pt')
rows=[json.loads(s) for s in (folder/'trajectory_complete.jsonl').read_text().splitlines()]
obs=torch.tensor(np.asarray([r['observation'] for r in rows]),dtype=torch.float32)
w=torch.tensor(np.asarray([r['preference'] for r in rows]),dtype=torch.float32)
r=np.asarray([x['feedback']['objective_vector'] for x in rows]);returns=np.zeros_like(r);carry=np.zeros(3)
for t in reversed(range(len(r))):
 if rows[t]['step']==c.horizon-1:carry[:]=0
 carry=r[t]+c.gamma*carry;returns[t]=carry
# 按完整回合留出，禁止随机拆步造成同日泄露。
episodes=np.asarray([x['episode'] for x in rows]);train=torch.tensor(episodes<6);held=~train
if not train.any() or not held.any():raise ValueError('要求至少八个完整训练回合')
y=torch.tensor(returns,dtype=torch.float32)
def metrics(mask):
 with torch.no_grad():
  pred=agent.value(obs[mask],w[mask]);target=y[mask];variance=target.var(0,unbiased=False)
  return dict(mse=((pred-target)**2).mean(0).tolist(),explained_variance=(1-(pred-target).var(0,unbiased=False)/(variance+1e-8)).tolist())
curve=[dict(steps=0,train=metrics(train),heldout=metrics(held))]
agent.update_statistics(y[train]);optimizer=torch.optim.Adam(agent.critic.parameters(),lr=c.lr)
for step in range(1,1001):
 prediction=agent.normalized_value(obs[train],w[train]);target=(y[train]-agent.return_mean)/agent.return_std
 loss=torch.nn.functional.smooth_l1_loss(prediction,target);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),.5);optimizer.step()
 if step in (5,20,100,200,1000):curve.append(dict(steps=step,train=metrics(train),heldout=metrics(held)))
result=dict(scope='离线critic可拟合性诊断；留出两完整训练日，不访问确认集，不更新actor；曲线不能证明控制提升',train_episodes=6,heldout_episodes=2,curve=curve)
Path('experiments/v44/critic_fit_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
