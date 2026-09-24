"""训练集岭回归残差与独立场景块校准；不把时序样本假定为独立。"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

TARGETS=np.array([1,3,6,7,8,9,10,11,12])
FEATURES=36
VERSION='ridge-block-conformal-physical-dt-v2'


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def design(x,mean,scale):
    z=np.clip((np.asarray(x)[...,:FEATURES]-mean)/scale,-10,10)
    return np.concatenate([np.ones((*z.shape[:-1],1)),z],axis=-1)


def fit(training,calibration,method='residual',alpha=.1,ridge=10.):
    if method not in ('hold','physics','residual'):raise ValueError('未知 DT 方法')
    train_ids={r['scenario'] for r in training};cal_ids={r['scenario'] for r in calibration}
    if not training or not calibration or train_ids&cal_ids:raise ValueError('训练与校准必须非空且场景互斥')
    if not 0<alpha<1 or ridge<=0:raise ValueError('alpha 或正则强度非法')
    x=np.array([r['x'] for r in training]);y=np.array([r['y'] for r in training])
    mean=x[:,:FEATURES].mean(0);scale=np.maximum(x[:,:FEATURES].std(0),.05)
    z=design(x,mean,scale);target=y-x[:,TARGETS]
    penalty=np.eye(z.shape[1])*ridge;penalty[0,0]=0
    coef=np.linalg.solve(z.T@z+penalty,z.T@target) if method=='residual' else np.zeros((z.shape[1],len(TARGETS)))
    model=dict(version=VERSION,method=method,mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),alpha=alpha,
        training_scenarios=sorted(train_ids),calibration_scenarios=sorted(cal_ids),training_hash=digest(training),calibration_hash=digest(calibration))
    # 每场景取所有时刻/输出维度的最大标准化误差，避免把同一天当作多条独立校准样本。
    train_err=y-predict(model,x)[0];output_scale=np.maximum(np.sqrt(np.mean(train_err**2,axis=0)),.005)
    model['scale_rule']='training_rmse_floor_0.005';model['output_scale']=output_scale.tolist()
    scores=[]
    for scene in sorted(cal_ids):
        part=[r for r in calibration if r['scenario']==scene]
        xx=np.array([r['x'] for r in part]);yy=np.array([r['y'] for r in part])
        scores.append(float(np.max(np.abs(yy-predict(model,xx)[0])/output_scale)))
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if rank>len(scores):raise ValueError('独立校准场景不足，当前覆盖率要求会产生无穷区间')
    q=sorted(scores)[rank-1];model.update(halfwidth=(q*output_scale).tolist(),calibration_rank=rank,calibration_blocks=len(scores))
    return model


def predict(model,x):
    a=np.asarray(x,dtype=float)
    if a.shape[-1]<FEATURES or not np.isfinite(a).all():raise ValueError('DT 输入维度或数值非法')
    mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    estimate=a[...,TARGETS]+design(a,mean,scale)@np.asarray(model['coef'])
    # 输出为归一化状态估计，不是已知车表；保持基本物理范围。
    price=estimate[...,6].copy();estimate=np.maximum(estimate,0);estimate[...,6]=price
    estimate[...,0]=np.clip(estimate[...,0],0,1)
    ood=np.max(np.abs((a[...,:FEATURES]-mean)/scale),axis=-1)>6
    return estimate,ood


def assess(model,records):
    ids={r['scenario'] for r in records}
    if ids&(set(model['training_scenarios'])|set(model['calibration_scenarios'])):raise ValueError('测试场景与拟合/校准场景重叠')
    x=np.array([r['x'] for r in records]);y=np.array([r['y'] for r in records]);p,ood=predict(model,x)
    inside=np.abs(p-y)<=np.asarray(model['halfwidth'])+1e-12
    blocks=[bool(np.all(inside[[i for i,r in enumerate(records) if r['scenario']==scene]])) for scene in sorted(ids)]
    return dict(rmse=float(np.sqrt(np.mean((p-y)**2))),per_target_rmse=np.sqrt(np.mean((p-y)**2,axis=0)).tolist(),
        marginal_coverage=float(inside.mean()),simultaneous_scenario_coverage=float(np.mean(blocks)),
        mean_interval_width=float(2*np.mean(model['halfwidth'])),ood_fraction=float(np.mean(ood)),scenarios=len(ids))


def save(model,path):Path(path).write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
