"""检查切换效用口径、共同任务序列和普通MAPPO负对照。"""
import argparse
import json
from pathlib import Path
import numpy as np


def verify(root):
    root=Path(root);out={}
    for method in ('ordinary','pareto'):
        data=json.loads((root/method/'results.json').read_text())['cases']
        assert len(data)==3 and all(not r['failed'] and r['completed'] and len(r['steps'])==24 for r in data)
        cases={r['mode']:r for r in data};reference=cases['switch']['steps']
        for r in data:
            assert all(s['requested']==reference[t]['requested'] for t,s in enumerate(r['steps']))
            assert all(np.isclose(s['utility'],np.dot(s['requested'],s['vector'])) for s in r['steps'])
            assert np.isclose(r['utility'],sum(s['utility'] for s in r['steps']))
        dynamic=np.array([s['action'] for s in reference]);fixed=np.array([s['action'] for s in cases['fixed_initial']['steps']])
        assert np.allclose(dynamic[:8],fixed[:8],atol=1e-8,rtol=0)
        distance=float(np.abs(dynamic[8:]-fixed[8:]).sum())
        if method=='ordinary':assert np.allclose(dynamic,fixed,atol=1e-8,rtol=0)
        out[method]=dict(cases=3,steps=72,post_switch_action_l1=distance,utilities={r['mode']:r['utility'] for r in data},
                         ac_violations=sum(s['ac_violations'] for r in data for s in r['steps']))
    out['interpretation']='2回合训练检查点的一日接口检查，不证明偏好学习收敛或切换优于固定策略'
    (root/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args();verify(a.root)
