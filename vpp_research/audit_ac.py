"""冻结训练策略后，独立验证执行前 AC 拒绝/修正机制。"""
import argparse
import json
from pathlib import Path
from .train import evaluate


def run(suite_folder):
    folder=Path(suite_folder);rows=[]
    for seed in range(1,6):
        for name,stress in [('nominal',None),('load_shock',{'load_scale':1.4})]:
            results=evaluate(folder/f'seed_{seed}'/'pareto_0/latest.pt',[[.2,.3,.5]],[9500,9501],robust=True,stress=stress,ac_safe=True)
            rows.append(dict(seed=seed,scenario=name,results=results))
    result=dict(scope='冻结策略的区间保护+现场 AC 校核；无已验证动作则仿真停止，非实际设备后备控制',rows=rows)
    (folder/'ac_verified_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite-folder',required=True);a=p.parse_args();run(a.suite_folder)
