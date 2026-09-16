"""按 DT 验证、偏好学习及真实曲线实验顺序执行的命令入口。"""
import argparse
from pathlib import Path
import json
from vpp_mappo.config import Config


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('dt');d.add_argument('--config',default='configs/research_smoke.json');d.add_argument('--output',required=True)
    s=sub.add_parser('suite');s.add_argument('--config',default='configs/research_smoke.json');s.add_argument('--dt-folder',required=True);s.add_argument('--output',required=True)
    s.add_argument('--joint',action='store_true',help='18 个能源/通信/计算智能体：普通与 Pareto 同组对照')
    s.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5]);s.add_argument('--episodes',type=int,default=12)
    r=sub.add_parser('real-data');r.add_argument('--config',default='configs/research_smoke.json');r.add_argument('--raw',required=True);r.add_argument('--output',required=True)
    a=p.parse_args();config=Config.load(a.config)
    if a.command=='dt':
        from .validate_dt import run
        result=run(config,a.output)
    elif a.command=='suite':
        from .suite import run
        if a.joint:config.cyber_mode='joint'
        result=run(config,a.dt_folder,a.output,tuple(a.seeds),a.episodes,families=('ordinary','pareto') if a.joint else None)
    else:
        from .real_data import run
        result=run(config,a.raw,a.output)
    print('完成；结果：',str(Path(a.output)/'results.json'))

if __name__=='__main__':main()
