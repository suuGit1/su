"""三目标 OLS 偏好搜索与官方固定权重 MAPPO 子策略。

按线性支持的顶点/乐观差距定义独立实现。RL 子问题近似求解，因此优先级不是理论上界证书。
参考 MORL-Baselines linear_support；不依赖其 cdd/cvxpy，也不替换普通 MAPPO。
"""
import itertools
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from .train import train, evaluate
from .suite import feasible


def corner_weights(values):
    v=np.asarray(values,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)==0 or not np.isfinite(v).all():
        raise ValueError('值向量必须为有限非空 N×3')
    # 四变量 (w1,w2,w3,u)，sum(w)=1，加三条独立激活边界确定顶点。
    a=np.vstack([np.c_[v,-np.ones(len(v))],np.c_[-np.eye(3),np.zeros(3)]])
    equality=np.array([1.,1.,1.,0.]);result=[]
    for ids in itertools.combinations(range(len(a)),3):
        matrix=np.vstack([equality,a[list(ids)]])
        if np.linalg.matrix_rank(matrix)<4:continue
        x=np.linalg.solve(matrix,[1.,0.,0.,0.])
        if np.max(a@x)>1e-7 or np.min(x[:3]) < -1e-7:continue
        w=np.maximum(0.,x[:3]);w/=w.sum()
        if not any(np.allclose(w,p,atol=1e-7,rtol=0) for p in result):result.append(w)
    return result


def next_weight(values, visited):
    for w in np.eye(3):
        if not any(np.allclose(w,p,atol=1e-7,rtol=0) for p in visited):return w, None
    v=np.asarray(values,dtype=float);weights=np.asarray(visited,dtype=float)
    if v.shape!=(len(weights),3):raise ValueError('每个访问权重必须对应一个验证向量')
    upper=np.max(weights@v.T,axis=1)
    candidates=[]
    for w in corner_weights(v):
        if any(np.allclose(w,p,atol=1e-7,rtol=0) for p in visited):continue
        lp=linprog(-w,A_ub=weights,b_ub=upper,bounds=[(None,None)]*3,method='highs')
        if not lp.success:raise RuntimeError('OLS 乐观优先级 LP 失败：'+lp.message)
        candidates.append((max(0.,float(-lp.fun-np.max(v@w))),w))
    if not candidates:return None,None
    priority,w=max(candidates,key=lambda x:x[0])
    return w,priority


def run(config, model, output, total_episodes=24, policies=6, reserve_mode='pcc_checked', validation_seeds=(8100,8101)):
    if policies<3 or total_episodes<policies or total_episodes%policies:
        raise ValueError('总训练回合需均分给至少三个子策略')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('OLS 输出目录必须为空')
    out.mkdir(parents=True,exist_ok=True)
    values=[];visited=[];members=[];per=total_episodes//policies
    def save(status):
        result=dict(protocol='ols-official-mappo-v1',reserve_mode=reserve_mode,status=status,
            planned_training_steps=total_episodes*config.horizon,
            completed_training_steps=sum(m['training_steps'] for m in members),
            completed_selection_steps=sum(m['selection_steps'] for m in members),
            budget_note='所有子策略训练均计入；自适应验证交互另记，论文比较还须匹配总交互或明确独立调参预算',
            priorities_are_certified_bounds=False,members=members)
        (out/'ols.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
    for i in range(policies):
        w,priority=next_weight(values,visited)
        if w is None:return save('no_unvisited_corner; unused_budget_not_spent')
        folder=out/f'member_{i}';cfg=replace(config,episodes=per)
        member=dict(index=i,weight=w.tolist(),priority=priority,training_steps=0,selection_steps=0,checkpoint=f'member_{i}/latest.pt')
        members.append(member);save('running')
        try:
            train(cfg,model,folder,method='fixed',preference=w.tolist(),scenario_offset=i*per,reserve_mode=reserve_mode)
            member['training_steps']=per*config.horizon
            rows=evaluate(folder/'latest.pt',[w],list(validation_seeds),ac_safe=True)
            # 成功回合完整计步；失败时需要逐步计数，不能凭失败回合推算为0。
            member['selection_steps']=sum(r.get('env_steps',config.horizon if not r['failed'] else 0) for r in rows)
            member['validation']=rows
            if len(rows)!=len(validation_seeds) or not all(feasible(r) for r in rows):
                save('validation_failed; excluded_from_support');raise RuntimeError('OLS 子策略未通过完整可行验证')
            value=np.mean([r['vector'] for r in rows],axis=0);member['validation_value']=value.tolist()
            values.append(value);visited.append(w);save('running')
        except (RuntimeError,ValueError) as exc:
            failure=folder/'failure.json'
            if failure.exists():member['training_steps']=json.loads(failure.read_text())['completed_env_steps']
            member['error']=str(exc);save('failed');raise
    return save('completed')


def select_member(result, preference):
    w=np.asarray(preference,dtype=float)
    if w.shape!=(3,) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1):
        raise ValueError('测试偏好须位于三维单纯形')
    valid=[m for m in result['members'] if 'validation_value' in m]
    if not valid:raise ValueError('没有可用的验证策略')
    # 选择仅依据验证集，禁止根据测试回报选择子策略。
    return max(valid,key=lambda m:float(w@np.asarray(m['validation_value'])))


if __name__=='__main__':
    import argparse
    from vpp_mappo.config import Config
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/research_smoke.json');p.add_argument('--dt-model',required=True)
    p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=24);p.add_argument('--policies',type=int,default=6)
    p.add_argument('--reserve-mode',choices=['linear','ac_checked','pcc_checked'],default='pcc_checked');a=p.parse_args()
    run(Config.load(a.config),json.loads(Path(a.dt_model).read_text()),a.output,a.episodes,a.policies,a.reserve_mode)
