"""按种子并行执行，同一DT与配置，完成后合并统一报告。"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import copy
import json
import multiprocessing
from pathlib import Path
from .real_campaign import parser,run,prepare_real,write
from .real_inputs import connect
from .real_report import report
from vpp_mappo.config import Config


def worker(a):
    run(a)
    return str(Path(a.output)/'results.json')


def main():
    p=parser();p.description=__doc__
    p.set_defaults(config='configs/v2_ieee33.json',output='runs/v2',episodes=12,eval_days=3,
        methods=['mpc','milp_oracle','ordinary','central','fixed','ols','pareto'])
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.workers<1:p.error('workers必须为正数；每个worker另使用cpu-threads线程')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists() and not a.resume and not a.dry_run:p.error('已有v2目录，请显式使用 --resume')
    # 先逐种子进行原入口预检，不在子进程中竞争创建DT缓存。
    jobs=[]
    for seed in a.seeds:
        child=copy.copy(a);child.seeds=[seed];child.output=str(root/f'seed_{seed}')
        child.dt_cache=str(root/'dt');child.dry_run=True
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):run(child)
        child.dry_run=a.dry_run;child.resume=True;jobs.append(child)
    plan=json.loads((Path(jobs[0].output)/'manifest.json').read_text());plan['seeds']=a.seeds
    write(root/'manifest.json',plan)
    if a.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    saved=plan['config'];known={f.name for f in fields(Config)}
    config=Config(**{k:v for k,v in saved.items() if k in known})
    for k,v in saved.items():
        if k.startswith('_'):setattr(config,k,v)
    _,paths,_,_=connect(a.data_root,config,ev_sessions=a.ev_sessions,dr_mode=a.dr_mode)
    prepare_real(config,paths,plan['protocol'],root/'dt',plan['dt_days'])
    print('v2 共享DT已准备；开始独立种子任务',flush=True)
    # HiGHS/PyTorch已初始化的线程锁不可通过fork继承；spawn创建干净的求解进程。
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        completed=list(pool.map(worker,jobs))
    entries=[e for path in completed for e in json.loads(Path(path).read_text())['entries']]
    write(root/'results.json',dict(manifest=plan,entries=entries,completed=True,
        all_successful=all(not e['failed'] and e['failed_or_infeasible']==0 and e.get('budget_complete',True) for e in entries)))
    report(root)
    write(root/'release.json',dict(version='2.0.0',objective_version='c3-ac-pcc-same-period-sampled-reserve-v4',
        dt_version='ridge-block-conformal-physical-dt-v2',seeds=a.seeds,workers=a.workers,
        note='按种子并行，子策略预算合计；训练完成不等于算法优势或AC鲁棒证明'))
    print('v2训练与报告完成：'+str(root),flush=True)
