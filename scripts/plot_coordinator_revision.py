"""展示每个日期的配对成本差，避免只展示有利的总体均值。"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    root=Path(root);r=json.loads((root/'summary.json').read_text())
    if not r['all_recorded'] or r['audit_issues']:raise ValueError('案例不完整或审计未通过')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    conditions=['nominal','uplink','compute','combined']
    for ax,baseline in zip(axes,['uniform','active_equal','joint_risk']):
        for revision,offset,color in [('mask_only',-.14,'#64748b'),('mapped',.14,'#2563eb')]:
            means=[]
            for i,cond in enumerate(conditions):
                row=r['contrasts'][f'{cond}/{revision}-v1_{baseline}']['cost']
                values=list(row['date_differences'].values());means.append(row['mean'])
                ax.scatter(np.full(len(values),i+offset),values,s=16,alpha=.5,color=color)
            ax.plot(np.arange(4)+offset,means,'D',color=color,label=revision)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),conditions,rotation=25)
        ax.set_title('Compared with v1 '+baseline);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Paired daily cost difference (lower is better)');axes[-1].legend()
    fig.suptitle('Coordinator revision — March development data; dots: dates, diamonds: means')
    fig.tight_layout();out=root/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'paired_cost.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');a=p.parse_args();plot(a.folder)
