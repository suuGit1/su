"""验收双 MAPPO 运行完整性和实际交互账本；通过不表示算法优越性。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.ledger import audit_training


def verify(folder):
    root=Path(folder).resolve()
    results=json.loads((root/'results.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    release=json.loads((root/'release.json').read_text(encoding='utf-8'))
    checks={'completed':results.get('completed') is True,'all_successful':results.get('all_successful') is True,
            'release_version':release.get('version')=='4.2.0'}
    entries=[]
    for seed in manifest['seeds']:
        for method in ('ordinary','pareto'):
            matches=[r for r in results['entries'] if r['seed']==seed and r['method']==method]
            key=f'{method}_{seed}'
            checks[key+'_unique_result']=len(matches)==1
            if len(matches)!=1:continue
            row=matches[0];folder=root/f'seed_{seed}'/key
            audit=audit_training(folder)
            training=[r for r in audit['runs'] if Path(r['path']).resolve()==folder]
            expected=manifest['episodes']*manifest['config']['horizon']
            checks[key+'_training_ledger']=len(training)==1 and training[0]['consistent'] and training[0]['committed_steps']==expected
            checks[key+'_test_complete']=not row['failed'] and row['failed_or_infeasible']==0 and row['test_steps']==manifest['eval_steps_per_method_seed'][method]
            entries.append(dict(seed=seed,method=method,training_audit=training,
                                selection_steps=row.get('selection_steps'),test_steps=row.get('test_steps'),
                                reserve_unconfirmed_steps=row.get('reserve_unconfirmed_steps')))
    record=dict(passed=all(checks.values()),checks=checks,entries=entries,
                result_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
                protocol=manifest['protocol'],
                scope='运行完整性验收；不证明学习收敛、统计优越性或连续AC鲁棒安全')
    (root/'acceptance_summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');a=p.parse_args()
    record=verify(a.folder);print(json.dumps({'passed':record['passed'],'checks':record['checks']},ensure_ascii=False,indent=2))
    sys.exit(0 if record['passed'] else 1)
