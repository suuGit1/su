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


def fit(training,calibration,method='residual',alpha=.1,ridge=10.,interval_mode='constant',scale_folds=5):
    if interval_mode not in ('constant','adaptive'):raise ValueError('未知区间模式')
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
    model['interval_mode']=interval_mode
    if interval_mode=='adaptive':
        if not isinstance(scale_folds,int) or scale_folds<2 or len(train_ids)<scale_folds:
            raise ValueError('自适应尺度需要至少 scale_folds 个训练日期，且折数至少2')
        # 按日期留折；尺度回归不读取校准标签。每折独立标准化，避免特征泄漏。
        ids=sorted(train_ids);errors=np.zeros_like(y)
        for fold in range(scale_folds):
            held=set(ids[fold::scale_folds]);mask=np.array([r['scenario'] in held for r in training])
            fm=x[~mask,:FEATURES].mean(0);fs=np.maximum(x[~mask,:FEATURES].std(0),.05)
            fz=design(x[~mask],fm,fs)
            fc=np.linalg.solve(fz.T@fz+penalty,fz.T@target[~mask]) if method=='residual' else np.zeros_like(coef)
            fp,_=predict(dict(mean=fm,scale=fs,coef=fc),x[mask])
            errors[mask]=np.abs(y[mask]-fp)
        # 强正则、固定截断范围；这些参数不根据测试结果选择。
        log_error=np.log(np.maximum(errors,.005))
        sp=np.eye(z.shape[1])*100.;sp[0,0]=0
        sc=np.linalg.solve(z.T@z+sp,z.T@log_error)
        model.update(scale_coef=sc.tolist(),scale_folds=scale_folds,
                     scale_rule='day_oof_log_abs_ridge100_floor0.005_v1')
    scores=[]
    for scene in sorted(cal_ids):
        part=[r for r in calibration if r['scenario']==scene]
        xx=np.array([r['x'] for r in part]);yy=np.array([r['y'] for r in part])
        scores.append(float(np.max(np.abs(yy-predict(model,xx)[0])/interval_scale(model,xx))))
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if rank>len(scores):raise ValueError('独立校准场景不足，当前覆盖率要求会产生无穷区间')
    q=sorted(scores)[rank-1];model.update(calibration_quantile=q,halfwidth=(q*output_scale).tolist(),calibration_rank=rank,calibration_blocks=len(scores))
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


def interval_scale(model,x):
    a=np.asarray(x,dtype=float)
    if a.shape[-1]<FEATURES or not np.isfinite(a).all():raise ValueError('DT 区间输入非法')
    if model.get('interval_mode','constant')=='adaptive':
        log_scale=design(a,model['mean'],model['scale'])@np.asarray(model['scale_coef'])
        base=np.maximum(np.asarray(model['output_scale']),.005)
        return np.clip(np.exp(np.clip(log_scale,-20,20)),.005,base*20)
    return np.broadcast_to(np.asarray(model['output_scale']),(*a.shape[:-1],len(TARGETS)))


def interval_halfwidth(model,x):
    # halfwidth 仅为兼容旧模型的参考宽度；在线自适应区间必须基于原始公开观察。
    if model.get('interval_mode','constant')=='adaptive':
        return model['calibration_quantile']*interval_scale(model,x)
    a=np.asarray(x)
    return np.broadcast_to(np.asarray(model['halfwidth']),(*a.shape[:-1],len(TARGETS)))


def assess(model,records):
    if not records:raise ValueError('测试记录为空')
    ids={r['scenario'] for r in records}
    if ids&(set(model['training_scenarios'])|set(model['calibration_scenarios'])):raise ValueError('测试场景与拟合/校准场景重叠')
    x=np.array([r['x'] for r in records]);y=np.array([r['y'] for r in records]);p,ood=predict(model,x)
    widths=interval_halfwidth(model,x)
    inside=np.abs(p-y)<=widths+1e-12
    blocks=[bool(np.all(inside[[i for i,r in enumerate(records) if r['scenario']==scene]])) for scene in sorted(ids)]
    return dict(rmse=float(np.sqrt(np.mean((p-y)**2))),per_target_rmse=np.sqrt(np.mean((p-y)**2,axis=0)).tolist(),
        marginal_coverage=float(inside.mean()),simultaneous_scenario_coverage=float(np.mean(blocks)),
        per_target_coverage=inside.mean(0).tolist(),per_target_width=(2*widths.mean(0)).tolist(),
        scenario_coverage=dict(zip(sorted(ids),blocks)),mean_interval_width=float(2*np.mean(widths)),ood_fraction=float(np.mean(ood)),scenarios=len(ids))


def save(model,path):Path(path).write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
