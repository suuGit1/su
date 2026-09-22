"""按完整场景延迟释放标签的滚动再校准；仅报告经验覆盖，不承诺漂移下名义保证。"""
import json
import math
from pathlib import Path
import numpy as np
from .dt import predict


class DelayedBlockCalibration:
    def __init__(self,scale,scores,alpha=.1,window=30):
        self.scale=np.asarray(scale,dtype=float);self.scores=list(scores);self.alpha=alpha;self.window=window;self.pending=[];self.clock=-float('inf')
        if self.scale.shape!=(9,) or not np.isfinite(self.scale).all() or (self.scale<=0).any():raise ValueError('校准尺度必须为九个有限正数')
        if not 0<alpha<1 or window<1 or not self.scores or not np.isfinite(self.scores).all() or min(self.scores)<0:raise ValueError('校准参数非法')

    def submit(self,score,available_at):
        if not np.isfinite([score,available_at]).all() or score<0:raise ValueError('场景分数/可用时刻非法')
        self.pending.append((available_at,float(score)))

    def widths(self,now):
        if now<self.clock:raise ValueError('再校准时间不能倒退')
        self.clock=now
        for t,s in sorted(self.pending):
            if t<=now:self.scores.append(s)
        self.pending=[x for x in self.pending if x[0]>now];self.scores=self.scores[-self.window:]
        rank=math.ceil((len(self.scores)+1)*(1-self.alpha))
        if rank>len(self.scores):raise ValueError('独立已到达校准块不足')
        return np.sort(self.scores)[rank-1]*self.scale


def replay(model,calibration,test,delay_blocks=1):
    if delay_blocks<1:raise ValueError('当前场景标签不能提前用于本场景')
    if {r['scenario'] for r in test}&({r['scenario'] for r in calibration}|set(model['training_scenarios'])):raise ValueError('时间划分重叠')
    scale=np.maximum(np.asarray(model['halfwidth']),1e-6)
    def error(records):
        x=np.array([r['x'] for r in records]);y=np.array([r['y'] for r in records]);return np.abs(predict(model,x)[0]-y)
    scores=[float(np.max(error([r for r in calibration if r['scenario']==s])/scale)) for s in sorted({r['scenario'] for r in calibration})]
    adaptive=DelayedBlockCalibration(scale,scores,model['alpha']);rows=[]
    for i,scene in enumerate(sorted({r['scenario'] for r in test})):
        width=adaptive.widths(i);err=error([r for r in test if r['scenario']==scene])
        rows.append(dict(scenario=scene,fixed_covered=bool(np.all(err<=scale)),adaptive_covered=bool(np.all(err<=width)),
            fixed_width=float(2*scale.mean()),adaptive_width=float(2*width.mean()),available_blocks=len(adaptive.scores)))
        adaptive.submit(float(np.max(err/scale)),i+delay_blocks)
    return dict(rows=rows,fixed_coverage=float(np.mean([r['fixed_covered'] for r in rows])),adaptive_coverage=float(np.mean([r['adaptive_covered'] for r in rows])),
        assumption='完整场景真值在之后的场景边界可用；仅离线时序回放，未连接现场真值采集',nominal_guarantee_under_shift=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    d=json.loads(Path(a.dataset).read_text())['physics'];m=json.loads(Path(a.model).read_text());Path(a.output).write_text(json.dumps(replay(m,d['calibration'],d['test']),ensure_ascii=False,indent=2))
