"""统一最大化约定的非支配筛选、精确小维度 HV 和参考集 IGD。"""
import numpy as np


def vectors(points, dimensions=3):
    x = np.asarray(points, dtype=float)
    if x.size == 0: return np.empty((0, dimensions))
    if x.ndim != 2 or x.shape[1] != dimensions or not np.isfinite(x).all():
        raise ValueError('目标必须为有限 N×D 数组')
    return x


def non_dominated(points, feasible=None):
    x = vectors(points)
    mask = np.ones(len(x),dtype=bool) if feasible is None else np.asarray(feasible,dtype=bool)
    if mask.shape != (len(x),): raise ValueError('可行性标记长度不匹配')
    selected=[]
    for i,p in enumerate(x):
        if not mask[i]: continue
        others=x[mask]
        dominated=np.any(np.all(others>=p,axis=1)&np.any(others>p,axis=1))
        duplicate=any(np.array_equal(p,x[j]) for j in selected)
        if not dominated and not duplicate: selected.append(i)
    return selected


def hypervolume(points, reference):
    ref=np.asarray(reference,dtype=float)
    if ref.shape != (3,) or not np.isfinite(ref).all(): raise ValueError('HV 参考点必须为三个有限数')
    x=vectors(points)
    # 只有完整支配参考点的盒子具有正体积；参考点由实验前统一固定。
    x=x[np.all(x>ref,axis=1)]
    def union_volume(p, origin):
        if len(p)==0: return 0.0
        if p.shape[1]==1: return float(max(0,p[:,0].max()-origin[0]))
        edges=np.unique(np.r_[origin[0],p[:,0]])
        total=0.0
        for left,right in zip(edges[:-1],edges[1:]):
            total+=(right-left)*union_volume(p[p[:,0]>=right,1:],origin[1:])
        return float(total)
    return union_volume(x,ref)


def igd(points, reference_front):
    x=vectors(points); ref=vectors(reference_front)
    if len(ref)==0 or len(x)==0: raise ValueError('IGD 必须提供非空参考前沿和候选集')
    return float(np.linalg.norm(ref[:,None,:]-x[None,:,:],axis=2).min(axis=1).mean())
