"""汇总安全降级与备用诊断；保持原始不可行日及HV筛选口径。"""
from collections import Counter
import json
from pathlib import Path


def report_v41(folder):
    root=Path(folder);data=json.loads((root/'results.json').read_text(encoding='utf-8'));entries=[]
    for entry in data['entries']:
        infos=[];missing=[]
        for row in entry.get('results',[]):
            path=Path(row['trace_path']) if row.get('trace_path') else None
            if path is None or not path.exists():missing.append(row.get('scenario_date'));continue
            infos.extend(json.loads(line)['feedback'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        entries.append(dict(method=entry['method'],seed=entry['seed'],steps=len(infos),missing_traces=missing,
            guard_reasons=dict(Counter(i.get('guard_reason','not_recorded') for i in infos)),
            reserve_status=dict(Counter(i.get('reserve_diagnostic',{}).get('status','not_recorded') for i in infos)),
            safety_modes=dict(Counter(i.get('safety_mode','not_recorded') for i in infos)),
            command_fallbacks=dict(Counter(i.get('command_fallback_reason','not_recorded') for i in infos)),
            solver_failures=dict(Counter(f['reason'] for i in infos for f in i.get('safety_failures',[]))),
            presolve_retries=sum(m.get('presolve_retry',False) for i in infos for m in i.get('reserve_diagnostic',{}).get('envelope',[])),
            certified_steps=sum(i['guard_certificate_survived'] for i in infos),reserve_invalid=sum(not i['reserve_valid'] for i in infos),
            hard_verified_steps=sum(i.get('safety_hard_verified',False) for i in infos),
            service_degraded_steps=sum(i.get('emergency_service_degraded',False) for i in infos)))
    result=dict(version='4.1.0',entries=entries,scope='区间不可行不缩窄；现场无违例不等于区间认证；MPC使用公开观察和自身发令历史，通信计算仍固定')
    (root/'v41_safety_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
