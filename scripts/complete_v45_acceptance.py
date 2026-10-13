"""汇总开发验收；结果直接写入项目目录，每个完整阶段立即生成独立归档。"""
import argparse,json,subprocess,sys,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='runs/v45_recovery');p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
archives=root/'archives';archives.mkdir(exist_ok=True)
def archive(name):
 folder=root/name
 with zipfile.ZipFile(archives/(name+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for f in folder.rglob('*'):
   if f.is_file() and f.name not in ('resume.pt','trajectory.jsonl') and f.suffix not in ('.tmp',):z.write(f,str(f.relative_to(root)))
def run(name,argv):
 if (root/name/'results.json').exists() or (name=='external_sample' and (root/name/'mpc/summary.json').exists() and json.loads((root/name/'mpc/summary.json').read_text()).get('completed')):
  archive(name);return
 with (root/(name+'.log')).open('a',encoding='utf-8') as log:
  r=subprocess.run([sys.executable,*map(str,argv)],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit(f'{name}失败；现场保留于{root/name}')
 archive(name);print(name,'完成且已归档',flush=True)
for group in ('safety','dt','network','coordinator'):
 run(group,['scripts/run_v45_mechanisms.py','--group',group,'--workers',a.workers,'--output',root/group])
run('coordinator_repaired',['scripts/run_v45_mechanisms.py','--group','coordinator','--priority-slack','.25','--workers',a.workers,'--output',root/'coordinator_repaired'])
run('external_sample',['scripts/run_v45_external_sample.py','--output',root/'external_sample'])
checkpoint=root/'diagnosis/conditioned/latest.pt';metadata=checkpoint.parent/'metadata.json';start=time.monotonic()
while not metadata.exists():
 if time.monotonic()-start>7200:raise SystemExit('诊断尚未完成，未启动依赖评估')
 time.sleep(5)
run('c3',['scripts/run_v45_mechanisms.py','--group','c3','--checkpoint',checkpoint,'--workers',a.workers,'--output',root/'c3'])
run('switch',['scripts/run_v45_switch.py','--checkpoint',checkpoint,'--csv',root/'diagnosis/development.csv','--days','2','--output',root/'switch'])
for name in ('diagnosis','seven'):
 if (root/name/'results.json').exists():archive(name)
print('机制与切换验收已归档；七基线是否完成另以其results.json为准',flush=True)
