"""真实数据统一 DT 准备、双 MAPPO 多种子训练与独立日期评价。"""
import argparse
import csv
from .real_benchmarks import learning,budgets
from .real_report import report
from dataclasses import replace
import json
from pathlib import Path
from .real_inputs import connect
from vpp_mappo.config import Config
from .dt import fit,assess,save,digest,VERSION as DT_VERSION
from .validate_dt import collect
from .train import train,evaluate


def parse_seeds(value):
    try:items=[int(s.strip()) for s in value.split(',')]
    except ValueError as exc:raise argparse.ArgumentTypeError('种子格式示例：1,2,3') from exc
    if not items or len(set(items))!=len(items) or min(items)<0:raise argparse.ArgumentTypeError('种子须唯一且非负')
    return items


def write(path,data):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)


def prepare_real(c,paths,protocol,dt,counts):
    dt=Path(dt);dt.mkdir(parents=True,exist_ok=True);key=digest(dict(protocol=protocol,days=counts,config=c.__dict__,dt_version=DT_VERSION))
    if (dt/'contract.json').exists() and json.loads((dt/'contract.json').read_text())['fingerprint']!=key:raise ValueError('DT 数据或配置发生变化')
    if (dt/'residual.json').exists():model=json.loads((dt/'residual.json').read_text())
    else:
        # 标签仅用于离线DT拟合；在线actor和MPC仍只读取受限公开观察。
        data={k:collect(c,range(30000,30000+n),'physics',paths[k]) for k,n in counts.items()}
        model=fit(data['train'],data['validation'],method='residual')
        model['real_input_fingerprint']=protocol['fingerprint']
        write(dt/'contract.json',dict(fingerprint=key));write(dt/'datasets.json',data);save(model,dt/'residual.json')
        print('真实数据DT训练与校准完成',flush=True)
    return model


def run(a):
    c,paths,sets,protocol=connect(a.data_root,Config.load(a.config),ev_sessions=a.ev_sessions,dr_mode=a.dr_mode)
    for key,value in [('lr',a.learning_rate),('ppo_epoch',a.ppo_epochs),('threads',a.cpu_threads),('solver_time_limit',a.solver_time_limit)]:
        if value is not None:setattr(c,key,value)
    c.validate()
    import numpy as np
    reference=getattr(a,'hv_reference',[-100.,-200.,0.])
    if len(reference)!=3 or not np.isfinite(reference).all():raise ValueError('HV参考点须包含三个有限数')
    if len(set(a.methods))!=len(a.methods):raise ValueError('算法列表不能重复')
    if a.episodes<1 or a.eval_days<1 or a.eval_days>len(sets['test'].profiles):
        raise ValueError(f'训练回合必须为正；独立测试只有 {len(sets["test"].profiles)} 天，禁止重复凑天数')
    counts={'train':len(sets['train'].profiles) if a.dt_train_days is None else a.dt_train_days,
            'validation':min(12,len(sets['validation'].profiles)-getattr(a,'selection_days',2)) if a.dt_calibration_days is None else a.dt_calibration_days}
    selection_days=getattr(a,'selection_days',2)
    if selection_days<1 or counts['validation']+selection_days>len(sets['validation'].profiles):raise ValueError('策略选择日期必须独立于DT校准，并且不能超过验证集长度')
    selection_dates=sets['validation'].scenario_names[counts['validation']:counts['validation']+selection_days]
    ols_policies=getattr(a,'ols_policies',4)
    if ols_policies<3:raise ValueError('OLS至少三个子策略')
    for m in a.methods:
        if m not in ('mpc','milp_oracle'):budgets(a.episodes,3 if m=='fixed' else ols_policies if m=='ols' else 1)
    if any(n<1 or n>len(sets[k].profiles) for k,n in counts.items()) or counts['validation']<9:
        raise ValueError('DT 天数超出数据范围，90%场景块校准至少需要9个独立日期')
    preferences=[]
    for part in a.eval_preferences.split(';'):
        w=[float(x) for x in part.split(',')]
        import numpy as np
        if len(w)!=3 or not np.isfinite(w).all() or min(w)<0 or not np.isclose(sum(w),1):raise ValueError('测试偏好非法')
        if any(np.allclose(w,old,atol=1e-9,rtol=0) for old in preferences):raise ValueError('测试偏好不能重复')
        preferences.append(w)
    plan=dict(config=c.__dict__,protocol=protocol,seeds=a.seeds,episodes=a.episodes,eval_days=a.eval_days,dt_days=counts,
        methods=a.methods,preferences=preferences,selection_days=selection_days,selection_dates=selection_dates,ols_policies=ols_policies,
        hv_reference=getattr(a,'hv_reference',[-100.,-200.,0.]),train_steps_per_method_seed={m:0 if m in ('mpc','milp_oracle') else a.episodes*c.horizon for m in a.methods},
        eval_steps_per_method_seed={m:a.eval_days*c.horizon*len(preferences) for m in a.methods},
        test_dates=sets['test'].scenario_names[:a.eval_days],
        note='训练回合可循环训练日期；eval-days为不重复测试日期；各测试偏好分别执行')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=plan:raise ValueError('输出目录协议不同，请使用新目录')
        if not a.resume and not a.dry_run:raise ValueError('已有实验目录，请显式使用 --resume')
    write(manifest,plan)
    if a.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    selection_dir=out/'selection';selection_dir.mkdir(exist_ok=True)
    selection_csv=selection_dir/'validation.csv'
    with paths['validation'].open() as source,selection_csv.open('w',newline='') as target:
        reader=csv.DictReader(source);writer=csv.DictWriter(target,reader.fieldnames);writer.writeheader()
        writer.writerows(r for r in reader if r['scenario'] in selection_dates)
    model=prepare_real(c,paths,protocol,getattr(a,'dt_cache',None) or out/'dt',counts)
    entries=[]
    for seed in a.seeds:
        for method in a.methods:
            dest=out/f'{method}_{seed}';result_path=out/f'{method}_{seed}.json'
            if result_path.exists():
                previous=json.loads(result_path.read_text())
                if not previous.get('failed') or not a.resume:
                    entries.append(previous);continue
                archive=dest/'failed_attempt_results';archive.mkdir(parents=True,exist_ok=True)
                result_path.rename(archive/f'{len(list(archive.iterdir()))+1}.json')
            cfg=replace(c,seed=seed,episodes=a.episodes)
            # dataclasses.replace 不复制动态快照；必须显式传入真实EV与DR协议。
            cfg._ev_bundle=c._ev_bundle;cfg._flex_record=c._flex_record
            row=dict(seed=seed,method=method,failed=False)
            try:
                if method in ('mpc','milp_oracle'):
                    from .integrated import rollout
                    def controller(csv_path,days,phase):
                        rows=[]
                        for wi,w in enumerate(preferences):
                            for day in range(days):
                                target=dest/phase/f'weight_{wi}_day_{day}'
                                saved=target/'summary.json'
                                if saved.exists() and json.loads(saved.read_text()).get('completed'):
                                    r=json.loads(saved.read_text())
                                else:
                                    if target.exists() and any(target.iterdir()):
                                        if not a.resume:raise ValueError('不完整控制器轨迹，请使用 --resume')
                                        index=1
                                        while target.with_name(target.name+f'_interrupted_{index}').exists():index+=1
                                        target.rename(target.with_name(target.name+f'_interrupted_{index}'))
                                    r=rollout(cfg,model,target,method=method,preference=w,seed=40000+day,csv_path=csv_path,ev_bundle=c._ev_bundle,profile_index=day)
                                rows.append(dict(preference=w,failed=not r['completed'],env_steps=r['steps'],
                                    vector=[-r['cost']/cfg.objective_scales[0],-r['carbon_kg']/cfg.objective_scales[1],r['reserve_kwh']/cfg.objective_scales[2]],
                                    violations=r['constraint_violations'],ac_violations=r['ac_violations'],ac_failed=0,
                                    reserve_invalid=r['reserve_unconfirmed_steps'],ev_unmet_kwh=r['ev_unmet_kwh'],dr_backlog_kwh=r['terminal_dr_kwh'],
                                    guard_infeasible=r['guard_infeasible'],guard_certified_steps=r['guard_certified_steps']))
                        return rows
                    validation=controller(selection_csv,selection_days,'validation')
                    results=controller(paths['test'],a.eval_days,'test')
                    from .suite import feasible
                    row.update(training_steps=0,budget_complete=True,selection_steps=sum(r['env_steps'] for r in validation),
                        validation=validation,results=results,test_steps=sum(r['env_steps'] for r in results),
                        reserve_unconfirmed_steps=sum(r['reserve_invalid'] for r in results),failed_or_infeasible=sum(not feasible(r) for r in results))
                else:
                    row.update(learning(cfg,model,dest,method,preferences,selection_csv,paths['test'],selection_days,a.eval_days,a.resume,ols_policies))
                row['test_dates']=plan['test_dates']
            except (RuntimeError,ValueError) as exc:
                row.update(failed=True,error=str(exc),budget_complete=False)
                # 即使评价失败，也不能把已经完成的训练交互记成零。
                completed=0
                for log in dest.rglob('attempts.jsonl'):
                    events=[json.loads(line) for line in log.read_text().splitlines()]
                    completed+=sum(e.get('event')=='step' for e in events)
                row['training_steps']=completed
            row['actual_training_steps']=sum(
                json.loads(line).get('event')=='step' for log in dest.rglob('attempts.jsonl') for line in log.read_text().splitlines())
            write(result_path,row);entries.append(row)
            write(out/'results.json',dict(manifest=plan,entries=entries,completed=False))
            print(seed,method,'完成' if not row['failed'] else row['error'],flush=True)
    write(out/'results.json',dict(manifest=plan,entries=entries,completed=True,
        all_successful=all(not r['failed'] and r['failed_or_infeasible']==0 and r.get('budget_complete',True) for r in entries),
        all_objectives_confirmed=all(not r['failed'] and r['failed_or_infeasible']==0 and r.get('reserve_unconfirmed_steps',0)==0 for r in entries)))

    report(out)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/connected_ieee33.json')
    p.add_argument('--learning-rate',type=float);p.add_argument('--ppo-epochs',type=int)
    p.add_argument('--cpu-threads',type=int);p.add_argument('--solver-time-limit',type=float)
    p.add_argument('--data-root',default='data/real');p.add_argument('--output',default='runs/real_campaign')
    p.add_argument('--seeds',type=parse_seeds,default=[1,2,3]);p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--eval-days',type=int,default=29)
    p.add_argument('--methods',nargs='+',choices=['mpc','milp_oracle','ordinary','pareto','central','fixed','ols'],default=['mpc','ordinary','pareto'])
    p.add_argument('--eval-preferences',default='0.2,0.3,0.5;0.6,0.1,0.3;0.1,0.7,0.2')
    p.add_argument('--dt-train-days',type=int);p.add_argument('--dt-calibration-days',type=int)
    p.add_argument('--selection-days',type=int,default=2)
    p.add_argument('--ols-policies',type=int,default=4)
    p.add_argument('--hv-reference',type=float,nargs=3,default=[-100.,-200.,0.])
    p.add_argument('--ev-sessions');p.add_argument('--dr-mode',choices=['off','sce-derated'],default='off')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--dt-cache',help='共享已准备的DT缓存；协议不匹配则拒绝')
    return p

def main():
    run(parser().parse_args())


if __name__=='__main__':main()
