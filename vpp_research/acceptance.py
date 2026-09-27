"""验收统一运行目录中的完整闭环证据。"""
import json
from pathlib import Path
import numpy as np


def verify(folder):
    root=Path(folder);checks={}
    for method,subdir in [('mpc','mpc'),('ordinary','ordinary_test'),('pareto','pareto_test')]:
        path=root/subdir
        config=json.loads((path/'config.json').read_text())
        summary=json.loads((path/'summary.json').read_text())
        rows=[json.loads(line) for line in (path/'trajectory.jsonl').read_text().splitlines()]
        checks[method]=dict(complete=summary['completed'] and len(rows)==config['horizon'],
            public_observation=all(np.asarray(r['observation']).shape==(18,65) for r in rows),
            coordinator=all(r['coordinator'] is not None and r['coordinator']['tasks'] for r in rows),
            dt_intervals=all(len(r['dt_halfwidth'])==9 for r in rows),
            c3_allocations=all(len(r['bandwidth_bps'])==len(r['cpu_cycles_per_second'])==6 for r in rows),
            safe_execution=all(r['feedback']['ac_violations']==0 and r['feedback']['constraint_violations']==0 for r in rows),
            service_delivery=all(r['feedback']['ev_unmet_kwh']<=1e-6 for r in rows) and abs(rows[-1]['feedback']['dr_backlog_kwh'])<=1e-6,
            pcc_objectives=all(r['feedback']['objective_version']=='c3-ac-pcc-same-period-sampled-reserve-v4' for r in rows))
    passed=all(all(row.values()) for row in checks.values())
    report=dict(passed=passed,checks=checks,scope='架构及当前场景功能验收，不证明任意扰动安全或算法优势')
    (root/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if not passed:raise RuntimeError('集成验收失败，参见 acceptance.json')
    return report
