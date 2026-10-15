"""由审计汇总生成可重绘研究图；开发集结果不得标为确认性结果。"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    root=Path(root);r=json.loads((root/'summary.json').read_text());s=r['summaries'];out=root/'figures';out.mkdir(exist_ok=True)
    names=['physics','residual'];a=np.full((2,2),np.nan)
    for i,c in enumerate(names):
        for j,w in enumerate(names):
            row=s.get(f'dt_factorial||{c}_{w}|');a[i,j]=row['mean_complete_day_cost'] if row else np.nan
    fig,ax=plt.subplots(figsize=(5.5,4.4));im=ax.imshow(a,cmap='YlGnBu');fig.colorbar(im,ax=ax,label='Mean daily cost (model units)')
    ax.set(xticks=[0,1],xticklabels=['Physics width','Residual width'],yticks=[0,1],yticklabels=['Physics center','Residual center'],title='DT factorial: March development dates')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',color='white' if a[i,j]>np.nanmean(a) else 'black')
    fig.tight_layout();fig.savefig(out/'dt_factorial.png',dpi=220);fig.savefig(out/'dt_factorial.pdf');plt.close(fig)
    conditions=['nominal','uplink','compute','combined'];variants=['uniform','active_equal','communication_age','computation_age','joint_age','joint_schedule','joint_risk']
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    for ax,cond in zip(axes.flat,conditions):
        values=[]
        for v in variants:
            m=s.get(f'c3|{cond}|{v}|');values.append(m['mean_complete_day_cost'] if m and m['completed']==m['requests'] else np.nan)
        ax.bar(range(len(variants)),values,color=['#334155','#64748b','#60a5fa','#93c5fd','#2563eb','#f59e0b','#ef4444'])
        ax.set_xticks(range(len(variants)),['All equal','Active equal','Comm','Compute','Joint','Schedule','Risk'],rotation=20)
        ax.set_title(cond);ax.set_ylabel('Mean daily cost (complete cells only)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed-capacity resource mechanisms — development data');fig.tight_layout()
    fig.savefig(out/'c3_cost.png',dpi=220);fig.savefig(out/'c3_cost.pdf');plt.close(fig)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');a=p.parse_args();print(plot(a.folder))
