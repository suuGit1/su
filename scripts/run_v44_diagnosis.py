"""相同学习器的三端点与条件策略诊断；只读训练集和独立二月开发日期。"""
import argparse,copy,csv,hashlib,json,sys,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_mappo.config import Config
from vpp_research.real_inputs import connect
from vpp_research.train_v4 import train_v4
from vpp_research.train import evaluate

def worker(task):
 c,model,folder,validation,days=task
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 train_v4(c,model,folder,method='pareto',preference=c.pareto_fixed_preference or (1/3,1/3,1/3),reserve_mode='pcc_checked',resume=(folder/'resume.pt').exists())
 rows=evaluate(folder/'latest.pt',[[1,0,0],[0,1,0],[0,0,1],[1/3]*3],range(50000,50000+days),csv_path=validation,trace_dir=folder/'diagnostic_evaluation')
 result=dict(case=folder.name,training=json.loads((folder/'budget_ledger.json').read_text()),results=rows)
 (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

def main(a):
 if a.episodes<4 or a.episodes%4:raise ValueError('回合数须为不小于4的4倍数，确保首批覆盖端点')
 if not 1<=a.eval_days<=2 or a.workers<1:raise ValueError('开发日期数为1或2，进程数必须正')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 c,paths,sets,source=connect(a.data_root,Config.load(a.config));c.episodes=a.episodes;c.seed=a.seed;c.validation_every=0
 with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:model=json.loads(z.read('residual.json'))
 dates=sets['validation'].scenario_names[25:27]
 if len(dates)!=2 or any(not x.startswith('2019-02-') for x in dates):raise ValueError('开发日期必须与DT校准分离')
 validation=root/'development.csv'
 with paths['validation'].open() as src,validation.open('w',newline='') as dst:
  reader=csv.DictReader(src);writer=csv.DictWriter(dst,reader.fieldnames);writer.writeheader();writer.writerows(r for r in reader if r['scenario'] in dates)
 cases={'conditioned':(), 'economic':(1.,0.,0.),'carbon':(0.,1.,0.),'reserve':(0.,0.,1.)}
 frozen=dict(arguments=vars(a),config=c.__dict__,data=source,dates=dates,model_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('vpp_mappo','vpp_research') for p in sorted(Path(d).rglob('*.py'))},runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 frozen=json.loads(json.dumps(frozen));fp=root/'frozen.json'
 if fp.exists() and json.loads(fp.read_text())!=frozen:raise ValueError('冻结协议改变，请使用新目录')
 fp.write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
 tasks=[]
 for name in a.cases:
  cfg=copy.deepcopy(c);cfg.pareto_fixed_preference=cases[name];cfg.validate();tasks.append((cfg,model,root/name,validation,a.eval_days))
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  results=list(pool.map(worker,tasks))
 (root/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print('诊断完成，各案例训练预算相同；尚不构成多种子收敛证据',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/v44_ieee33.json');p.add_argument('--data-root',default='data/real');p.add_argument('--episodes',type=int,default=8);p.add_argument('--seed',type=int,default=43);p.add_argument('--eval-days',type=int,default=1);p.add_argument('--workers',type=int,default=4);p.add_argument('--cases',nargs='+',choices=['conditioned','economic','carbon','reserve'],default=['conditioned','economic','carbon','reserve']);p.add_argument('--output',default='runs/v44_diagnosis');main(p.parse_args())
