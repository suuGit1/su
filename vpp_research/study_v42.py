"""真实能源数据 DT 配对研究。日期为统计单位；保留失败，不用种子复制日期凑样本。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.observed_mpc import ObservedMPC
from .real_inputs import connect
from .real_campaign import write
from .validate_dt import collect
from .dt import fit, assess, save, digest
from .environment import ResearchEnv
from .suite import stats


def control_case(task):
    c,csv_path,model,index,date,seed,out,name=task
    file=out/f'control_{index:02d}_{name}.json'
    if file.exists():row=json.loads(file.read_text())
    else:
        row=dict(date=date,method=name,seed=seed+index,failed=False,steps=[])
        started=time.perf_counter()
        try:
            env=ResearchEnv(c,csv_path,dt_model=model,reserve_mode='pcc_checked')
            obs,_=env.reset(seed+index,index);planner=ObservedMPC(c,env.spec,env.flex,2)
            for step in range(c.horizon):
                action,meta=planner.propose(obs[0]);obs,_,_,done,info=env.step_candidate(action,np.ones(6),np.ones(6))
                row['steps'].append(dict(step=step,cost=float(info['objective_cost']+info['terminal_penalty']),
                    mse=float(info['research_dt_mse']),covered=bool(info['interval_covered']),
                    certificate=bool(info['guard_certificate_survived']),ac_violations=int(info['ac_violations']),
                    reserve_valid=bool(info['reserve_valid']),mpc_failed=bool(meta['mpc_failed'])))
            row.update(cost=sum(s['cost'] for s in row['steps']),completed=bool(done))
        except (RuntimeError,ValueError) as exc:
            row.update(failed=True,error=str(exc),reason=getattr(exc,'reason',None))
        row['elapsed_seconds']=time.perf_counter()-started;write(file,row)
    return row


def run(a):
    if a.workers<1:raise ValueError('workers必须为正')
    c,paths,sets,source=connect(a.data_root,Config.load(a.config))
    if not 9<=a.calibration_days<=len(sets['validation'].profiles)-2:raise ValueError('校准至少9天，另外保留2天策略选择')
    if not 1<=a.eval_days<=len(sets['test'].profiles):raise ValueError('测试日期数非法')
    if not 0<=a.control_days<=a.eval_days:raise ValueError('闭环日期数不能超过测试日期数')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(source=source,config=c.__dict__,calibration_days=a.calibration_days,eval_days=a.eval_days,
                  control_days=a.control_days,seed=a.seed,alpha=.1,scale_folds=5,scale_ridge=100,
                  methods=['physics','residual','adaptive'],
                  limits=['March dates were used in prior development: retrospective evaluation, not untouched confirmatory holdout',
                          'Real GB energy/carbon curves; IEEE33, devices and cyber parameters are simulated assumptions',
                          'EV and DR disabled without matched real sessions/parameters',
                          'Day exchangeability is not established; coverage and confidence intervals are empirical/descriptive',
                          'Paired cost statistics condition on both methods completing; failures are reported separately'])
    protocol=json.loads(json.dumps(protocol))
    contract=out/'protocol.json'
    if contract.exists() and json.loads(contract.read_text())!=protocol:raise ValueError('输出目录协议不一致，请使用新目录')
    write(contract,protocol) # 在读取测试标签与运行前冻结协议
    counts=dict(train=len(sets['train'].profiles),validation=a.calibration_days,test=a.eval_days)
    data={}
    for split,n in counts.items():
        file=out/(split+'_records.json')
        if file.exists():data[split]=json.loads(file.read_text())
        else:
            data[split]=collect(c,range(a.seed,a.seed+n),'physics',paths[split]);write(file,data[split])
        print('collected',split,n,flush=True)
    models={name:fit(data['train'],data['validation'],method='physics' if name=='physics' else 'residual',
                     interval_mode='adaptive' if name=='adaptive' else 'constant') for name in protocol['methods']}
    for name,model in models.items():save(model,out/(name+'.json'))
    result=dict(protocol_hash=digest(protocol),estimation={name:assess(model,data['test']) for name,model in models.items()},control=[])
    write(out/'results.json',result)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    tasks=[(c,paths['test'],model,index,sets['test'].scenario_names[index],a.seed,out,name)
           for index in range(a.control_days) for name,model in models.items()]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(control_case,tasks):
            result['control'].append(row)
            write(out/'results.json',result)
            print('control',row['date'],row['method'],'failed',row['failed'],'steps',len(row['steps']),flush=True)
    result['paired_cost']={}
    for left,right in [('residual','physics'),('adaptive','residual'),('adaptive','physics')]:
        deltas=[];failed=0
        for date in sets['test'].scenario_names[:a.control_days]:
            pair={r['method']:r for r in result['control'] if r['date']==date}
            if pair[left]['failed'] or pair[right]['failed']:failed+=1
            else:deltas.append(pair[left]['cost']-pair[right]['cost'])
        result['paired_cost'][left+'_minus_'+right]=dict(summary=stats(deltas) if deltas else None,failed_pairs=failed)
    write(out/'results.json',result)
    from .report_v42 import report
    report(out)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/real');p.add_argument('--config',default='configs/v42_adaptive_ieee33.json')
    p.add_argument('--output',required=True);p.add_argument('--calibration-days',type=int,default=25)
    p.add_argument('--eval-days',type=int,default=29);p.add_argument('--control-days',type=int,default=29)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=42000)
    run(p.parse_args())

if __name__=='__main__':main()
