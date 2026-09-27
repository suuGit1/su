"""逐步轨迹复算与完整性核验，失败和缺失证据保持可见。"""
import json
from pathlib import Path
import numpy as np


def audit_trace(path, expected=None):
    path=Path(path)
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    infos=[r['feedback'] for r in records]
    vector=np.sum([i['objective_vector'] for i in infos],axis=0).tolist() if infos else [0.,0.,0.]
    summary=dict(env_steps=len(records),vector=vector,
        violations=sum(i['constraint_violations'] for i in infos),ac_violations=sum(i['ac_violations'] for i in infos),
        reserve_invalid=sum(not i['reserve_valid'] for i in infos),
        guard_certified_steps=sum(i['guard_certificate_survived'] for i in infos),
        ev_unmet_kwh=sum(i['ev_unmet_kwh'] for i in infos),dr_backlog_kwh=infos[-1]['dr_backlog_kwh'] if infos else 0.)
    errors=[]
    ledger_path=path.parent/'evaluation.sqlite'
    if ledger_path.exists():
        from .ledger import Ledger
        ledger=Ledger(ledger_path);counts=ledger.summary();ledger.close()
        if counts['successful_calls']!=len(records) or counts['unknown_calls'] or counts['error_calls']:errors.append('评价账本与轨迹不一致或存在失败/未知调用')
    else:errors.append('缺少v3评价账本')
    if [r.get('step') for r in records]!=list(range(len(records))):errors.append('时序不连续或重复')
    if expected:
        for key in summary:
            if key in expected and expected[key] is not None and not np.allclose(summary[key],expected[key],rtol=1e-7,atol=1e-7):errors.append(key+'复算不一致')
        if expected.get('failed'):errors.append('场景失败')
    return dict(path=str(path),consistent=bool(records) and not errors,errors=errors,recomputed=summary)


def audit_campaign(root):
    root=Path(root);data=json.loads((root/'results.json').read_text());entries=[]
    for e in data['entries']:
        traces=[]
        for phase in ('validation','results'):
            for r in e.get(phase,[]):
                if r.get('trace_path'):a=audit_trace(r['trace_path'],r)
                else:a=dict(consistent=False,errors=['缺少轨迹路径'])
                traces.append(dict(phase=phase,**a))
        entries.append(dict(method=e['method'],seed=e['seed'],failed=e.get('failed',False),traces=traces,
                            consistent=bool(traces) and not e.get('failed',False) and all(t['consistent'] for t in traces)))
    from .ledger import Ledger
    budgets={}
    for path in sorted(root.rglob('evaluation.sqlite')):
        phase='test' if 'test' in path.parts else 'validation' if 'validation' in path.parts else 'other'
        ledger=Ledger(path);counts=ledger.summary();ledger.close()
        target=budgets.setdefault(phase,dict(attempted_calls=0,successful_calls=0,error_calls=0,unknown_calls=0))
        for key in target:target[key]+=counts[key]
    result=dict(schema='trace-evidence-v3',entries=entries,all_attempt_evaluation_budget=budgets,
                consistent=bool(entries) and all(e['consistent'] for e in entries))
    (root/'trace_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
