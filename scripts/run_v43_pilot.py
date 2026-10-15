"""v4.3真实数据小预算端到端验收；训练与末端选模共同计入预算。"""
import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_mappo.config import Config
from vpp_research.real_inputs import connect
from vpp_research.real_benchmarks import learning


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def worker(task):
    cfg,model,selection,test,method,seed,training,total,days,root=task
    cfg=copy.deepcopy(cfg);cfg.seed=seed;cfg.episodes=training;cfg.validate()
    folder=root/f'{method}_{seed}';result_path=root/f'{method}_{seed}.json'
    if result_path.exists():return json.loads(result_path.read_text())
    result=dict(method=method,seed=seed,failed=False,total_budget_steps=total*cfg.horizon)
    try:
        result.update(learning(cfg,model,folder,method,[[.2,.3,.5],[.6,.1,.3],[.1,.7,.2]],selection,test,2,days,resume=True,ols_policies=4))
        train_ledgers=[json.loads(p.read_text()) for p in folder.rglob('budget_ledger.json')]
        validation_ledgers=[json.loads(p.read_text()).get('interaction_ledger',{}) for p in folder.rglob('evaluation.json') if 'validation' in p.parts]
        ledgers=train_ledgers+validation_ledgers
        result['attempted_train_selection_steps']=sum(r.get('attempted_calls',0) for r in ledgers)
        result['ledger_failures']=sum(r.get('unknown_calls',0)+r.get('error_calls',0) for r in ledgers)
        result['budget_matched']=bool(train_ledgers) and all('attempted_calls' in r for r in ledgers) and result['attempted_train_selection_steps']==total*cfg.horizon and result['ledger_failures']==0
    except (RuntimeError,ValueError) as exc:result.update(failed=True,error=str(exc))
    write(result_path,result);return result


def run(a):
    c,paths,sets,source=connect(a.data_root,Config.load('configs/v43_ieee33.json'))
    if any(not d.startswith('2019-03-') for d in sets['test'].scenario_names):raise ValueError('仅限三月开发集')
    if a.workers<1 or not 1<=a.eval_days<=len(sets['test'].profiles):raise ValueError('进程数或评估日数非法')
    seeds=[int(v) for v in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or min(seeds)<0:raise ValueError('种子须非负且不重复')
    if len(set(a.methods))!=len(a.methods):raise ValueError('方法不可重复')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:model=json.loads(z.read('residual.json'))
    dates=sets['validation'].scenario_names[25:27]
    if len(dates)!=2:raise ValueError('需要与25日DT校准分开的2日选择集')
    selection=root/'selection.csv'
    with paths['validation'].open() as source_file,selection.open('w',newline='') as target:
        reader=csv.DictReader(source_file);writer=csv.DictWriter(target,reader.fieldnames);writer.writeheader();writer.writerows(r for r in reader if r['scenario'] in dates)
    allocations={}
    for method in a.methods:
        members=3 if method=='fixed' else 4 if method=='ols' else 1
        selection_episodes=(4 if method=='pareto' else members)*2
        training=a.total_episodes-selection_episodes
        if training<members:raise ValueError('预算不足')
        allocations[method]=dict(training_episodes=training,selection_episodes=selection_episodes)
    frozen=dict(config=c.__dict__,seeds=seeds,total_episodes=a.total_episodes,eval_days=a.eval_days,allocations=allocations,
                data=source,selection_dates=dates,dt_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in ('vpp_research','vpp_mappo') for p in sorted(Path(name).rglob('*.py'))})
    frozen=json.loads(json.dumps(frozen));path=root/'frozen.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('冻结配置已变更，请使用新目录')
    write(path,frozen)
    jobs=[(c,model,selection,paths['test'],m,s,allocations[m]['training_episodes'],a.total_episodes,a.eval_days,root) for s in seeds for m in a.methods]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[]
        for r in pool.map(worker,jobs):rows.append(r);print(r['method'],r['seed'],'failed',r['failed'],'budget_matched',r.get('budget_matched'),flush=True)
    write(root/'results.json',dict(entries=rows,scope='小预算集成验收，不证明收敛或算法优势'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='data/real');p.add_argument('--seeds',default='43');p.add_argument('--methods',nargs='+',choices=['ordinary','pareto','fixed','ols','central'],default=['ordinary','pareto','fixed','ols']);p.add_argument('--total-episodes',type=int,default=12);p.add_argument('--eval-days',type=int,default=1);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',default='runs/v43_pilot');run(p.parse_args())
