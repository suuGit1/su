"""导出可复现科学图；只绘制实际报告中的数据，不平滑或补造实验。"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    root=Path(folder);out=root/'figures';out.mkdir(exist_ok=True)
    report=json.loads((root/'report.json').read_text());rows=report['rows']
    methods=sorted({r['method'] for r in rows})
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for i,method in enumerate(methods):
        values=[r['hv'] for r in rows if r['method']==method]
        ax.scatter(np.full(len(values),i),values,label=method,s=32)
        ax.plot([i-.2,i+.2],[np.mean(values)]*2,color='black')
    ax.set_xticks(range(len(methods)),methods,rotation=20);ax.set_ylabel('Hypervolume')
    ax.set_title('Per-seed feasible fronts (bar: mean; empty front: zero)')
    fig.savefig(out/'hypervolume.png',dpi=180);fig.savefig(out/'hypervolume.svg');plt.close(fig)
    fig=plt.figure(figsize=(8,6),layout='constrained');ax=fig.add_subplot(projection='3d')
    for method in methods:
        points=[p for r in rows if r['method']==method for p in r['front']]
        if points:
            a=np.array(points);ax.scatter(-a[:,0],-a[:,1],a[:,2],label=method)
    ax.set_xlabel('Cost / 100 (min)');ax.set_ylabel('Carbon / 100 (min)');ax.set_zlabel('Reserve / 100 (max)')
    if ax.collections:ax.legend()
    ax.set_title('Feasible non-dominated points per method and seed')
    fig.savefig(out/'pareto.png',dpi=180);fig.savefig(out/'pareto.svg');plt.close(fig)
    if (root/'dt_validation.json').exists():
        d=json.loads((root/'dt_validation.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        days=range(len(d['physics']['closed_loop']))
        for name in ('physics','residual'):
            entries=d[name]['closed_loop']
            axes[0].plot(days,[r['ac_cost'] for r in entries],marker='o',label=name)
            axes[1].plot(days,[r['guard_certified_steps'] for r in entries],marker='o',label=name)
        axes[0].set_ylabel('Daily AC control cost');axes[1].set_ylabel('Certified steps / day')
        for ax in axes:ax.set_xlabel('Independent test day');ax.set_xticks(list(days));ax.legend()
        fig.savefig(out/'dt_control.png',dpi=180);fig.savefig(out/'dt_control.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');plot(p.parse_args().folder)
