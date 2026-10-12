"""生成训练＋选模总预算相等的真实数据实验命令；默认只生成计划。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def build(a):
    budgets=[int(v) for v in a.budgets.split(',')];seeds=[int(v) for v in a.seeds.split(',')]
    if not budgets or len(set(budgets))!=len(budgets) or min(budgets)<12 or len(set(seeds))!=len(seeds) or not seeds or min(seeds)<0 or a.workers<1:raise ValueError('预算、种子或进程数非法')
    methods=['ordinary','pareto','fixed','ols','central'];jobs=[]
    for budget in budgets:
        for method in methods:
            members=3 if method=='fixed' else 4 if method=='ols' else 1
            selection=(4 if method=='pareto' else members)*2
            training=budget-selection
            if training<members:raise ValueError('总预算不足以覆盖训练与选择')
            output=str(Path(a.output)/f'budget_{budget}'/method)
            command=[sys.executable,'run_v42.py','--data-root',a.data_root,'--seeds',a.seeds,'--episodes',str(training),
                     '--selection-days','2','--eval-days','29','--dt-train-days','31','--dt-calibration-days','25',
                     '--validation-every','0','--ols-policies','4','--methods',method,'--eval-preferences',a.preferences,
                     '--workers',str(a.workers),'--cpu-threads','1','--output',output]
            jobs.append(dict(budget_episodes=budget,method=method,members=members,training_episodes=training,
                             selection_episodes=selection,total_steps_per_seed=budget*24,command=command,output=output))
    jobs.append(dict(method='model_based',command=[sys.executable,'run_v42.py','--data-root',a.data_root,'--seeds',a.seeds,
        '--episodes','4','--eval-days','29','--selection-days','2','--dt-train-days','31','--dt-calibration-days','25',
        '--methods','mpc','milp_oracle','--eval-preferences',a.preferences,'--workers',str(a.workers),'--output',str(Path(a.output)/'model_based')],
        output=str(Path(a.output)/'model_based'),reuse='各学习预算共享同一模型对照结果；不计作重新执行') )
    return dict(protocol='real-train-plus-selection-v1',seeds=seeds,budgets=budgets,horizon=24,jobs=jobs,
                scope='三月开发集学习曲线；未解封确认集；v4.2原协调器，候选服务协调器不得宣称已集成训练',
                validation_every=0,hv_reference=[-100.,-200.,0.],config_sha256=hashlib.sha256(Path('configs/v42_ieee33.json').read_bytes()).hexdigest(),
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('vpp_research','vpp_mappo') for p in sorted(Path(folder).rglob('*.py'))},
                limits=['预算阶梯独立重训，所有阶梯成本累加，不是免费续训','预算充分性须由学习曲线判断，不预先宣布收敛','OLS提前停止则标为预算未完成，不替它补造交互','选模仅末端，禁用未计账周期验证；实际结果必须再次审计','测试交互单独计账，Oracle使用未来信息单列'])


def run(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);plan=build(a);path=root/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('计划已冻结，请使用新目录')
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    if not a.execute:print('计划已生成，尚未启动训练：'+str(path));return
    audits=[]
    for job in plan['jobs']:
        command=job['command']+(['--resume'] if (Path(job['output'])/'manifest.json').exists() else [])
        subprocess.run(command,check=True)
        results=json.loads((Path(job['output'])/'results.json').read_text())
        for r in results['entries']:
            actual=r.get('training_steps',0)+r.get('selection_steps',0)
            folder=Path(job['output'])/f"seed_{r['seed']}"/f"{r['method']}_{r['seed']}"
            ledgers=[json.loads(p.read_text()) for p in folder.rglob('budget_ledger.json')]
            evaluations=[json.loads(p.read_text()).get('interaction_ledger',{}) for p in folder.rglob('evaluation.json') if 'validation' in p.parts]
            selection_dbs=[p for p in folder.rglob('evaluation.sqlite') if 'validation' in p.parts]
            evidence=ledgers+evaluations
            attempted=sum(v.get('attempted_calls',0) for v in evidence)
            errors=sum(v.get('error_calls',0)+v.get('unknown_calls',0) for v in evidence)
            verified=bool(ledgers) and len(selection_dbs)==len(evaluations) and all('attempted_calls' in v for v in evidence)
            audits.append(dict(ledger_attempted_train_selection_steps=attempted,ledger_verified=verified,
                ledger_error_or_pending_calls=errors,ledger_budget_matched=verified and errors==0 and attempted==job.get('total_steps_per_seed'),method=r['method'],seed=r['seed'],failed=r['failed'],budget=job.get('total_steps_per_seed'),
                actual_train_selection_steps=actual,budget_matched=actual==job.get('total_steps_per_seed') if 'total_steps_per_seed' in job else None))
        (root/'budget_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    print('执行完成；须检查实际预算匹配及失败，不能由计划预算替代实际预算')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--budgets',default='300,1200,4800');p.add_argument('--seeds',default='1,2,3,4,5');p.add_argument('--workers',type=int,default=1);p.add_argument('--data-root',default='data/real');p.add_argument('--output',default='runs/phase3_plan');p.add_argument('--preferences',default='0.2,0.3,0.5;0.6,0.1,0.3;0.1,0.7,0.2');p.add_argument('--execute',action='store_true');run(p.parse_args())
