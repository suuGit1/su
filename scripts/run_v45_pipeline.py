"""P0–P4统一命令计划及顺序执行；预算开放配置，运行失败即停止并保留现场。"""
import argparse,json,subprocess,sys,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def commands(a):
    root=Path(a.output);jobs=[];py=sys.executable
    def add(stage,args):
        if stage in a.stages:jobs.append(dict(stage=stage,argv=[py,*map(str,args)]))
    add('P0',['scripts/test_terminal_v45.py'])
    add('P0',['scripts/test_ledger_journal.py'])
    add('P0',['scripts/test_pareto_v44.py'])
    add('P0',['scripts/run_v45_mechanisms.py','--group','safety','--workers',a.workers,'--output',root/'safety'])
    for seed in a.seeds.split(','):
        dest=root/f'diagnosis_{seed}'
        add('P1',['scripts/run_v44_diagnosis.py','--config',a.config,'--data-root',a.data_root,'--seed',seed,'--episodes',a.diagnosis_episodes,'--eval-days',min(2,a.eval_days),'--workers',a.workers,'--output',dest])
        add('P1',['scripts/report_v44_diagnosis.py',dest,dest/'report.json'])
    checkpoint=root/f"diagnosis_{a.seeds.split(',')[0]}"/'conditioned/latest.pt'
    for group in ('dt','c3','coordinator'):
        args=['scripts/run_v45_mechanisms.py','--group',group,'--days',min(2,a.eval_days),'--workers',a.workers,'--output',root/group]
        if group!='dt':args+=['--checkpoint',checkpoint]
        add('P2',args)
    add('P3',['scripts/run_v45_switch.py','--checkpoint',checkpoint,'--csv',root/f"diagnosis_{a.seeds.split(',')[0]}"/'development.csv','--days',min(2,a.eval_days),'--output',root/'switch'])
    add('P3',['run_v45.py','--config',a.config,'--data-root',a.data_root,'--seeds',a.seeds,'--episodes',a.episodes,'--eval-days',a.eval_days,'--selection-days',a.selection_days,'--workers',a.workers,'--output',root/'fair'])
    add('P4',['scripts/run_v45_mechanisms.py','--group','network','--workers',a.workers,'--output',root/'network'])
    add('P4',['scripts/run_v45_external_sample.py','--config',a.config,'--data-root',a.data_root,'--output',root/'external_sample'])
    add('P4',['scripts/audit_v45_data.py','--data-root',a.data_root,'--output',root/'data_audit.json'])
    for job in jobs:
        if job['argv'][1]=='scripts/run_v45_mechanisms.py':job['argv']+=['--data-root',a.data_root,'--config',a.config]
    jobs.append(dict(stage='report',argv=[py,'scripts/report_v45.py',str(root)]))
    return jobs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stages',nargs='+',choices=['P0','P1','P2','P3','P4'],default=['P0','P1','P2','P3','P4']);p.add_argument('--data-root',default='data/real');p.add_argument('--config',default='configs/v45_ieee33.json');p.add_argument('--ledger-backend',choices=['sqlite','journal'],default='sqlite');p.add_argument('--seeds',default='1,2,3');p.add_argument('--diagnosis-episodes',type=int,default=32);p.add_argument('--episodes',type=int,default=120);p.add_argument('--eval-days',type=int,default=29);p.add_argument('--selection-days',type=int,default=2);p.add_argument('--workers',type=int,default=1);p.add_argument('--output',default='runs/v45_pipeline');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if min(a.episodes,a.eval_days,a.selection_days,a.workers)<1 or a.diagnosis_episodes<4 or a.diagnosis_episodes%4:p.error('预算或进程参数非法')
    seeds=[int(s) for s in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or min(seeds)<0:p.error('种子必须非负且唯一')
    a.output=str(Path(a.output).resolve());root=Path(a.output);root.mkdir(parents=True,exist_ok=True);jobs=commands(a)
    plan=dict(arguments=vars(a),jobs=jobs,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('vpp_mappo','vpp_research','scripts') for p in sorted((ROOT/folder).rglob('*.py'))},scope='各端点等单策略预算诊断与等家族训练预算实验分开；P4未自动获得完整EV或硬件功耗；不解封确认集')
    path=root/'pipeline.json'
    if path.exists():
        old=json.loads(path.read_text());old['arguments']['execute']=a.execute
        if old!=plan:raise ValueError('流水线计划已改变，使用新目录')
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    if not a.execute:print(json.dumps(plan,ensure_ascii=False,indent=2))
    else:
        env=os.environ.copy();env['VPP_LEDGER_JOURNAL']='1' if a.ledger_backend=='journal' else '0'
        for i,job in enumerate(jobs):
            receipt=root/f'job_{i:03d}.json'
            if receipt.exists() and json.loads(receipt.read_text())['returncode']==0:continue
            print(job['stage'],subprocess.list2cmdline(job['argv']),flush=True)
            with (root/f'job_{i:03d}.log').open('a',encoding='utf-8') as f:r=subprocess.run(job['argv'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            receipt.write_text(json.dumps(dict(**job,returncode=r.returncode),ensure_ascii=False,indent=2))
            if r.returncode:raise SystemExit('阶段失败，完整日志已保存；禁止跳过失败声称全部完成')
