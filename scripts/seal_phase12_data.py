"""仅做完整性检查并封存新月份；不评估策略，不重新拟合缩放。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from vpp_research.public_data import COLUMNS,download_carbon


def seal(root):
    root=Path(root);out=root/'confirmation_2019Q2';out.mkdir(exist_ok=True)
    if (out/'seal.json').exists():
        m=json.loads((out/'seal.json').read_text())
        if hashlib.sha256((out/'sealed.csv').read_bytes()).hexdigest()!=m['sha256']:raise ValueError('封存文件已改变')
        return m
    sources=download_carbon(out/'carbon_raw','2019-03-31','2019-07-01')
    records=[]
    for item in sources:
        for r in json.loads((out/'carbon_raw'/ (item['date']+'.json')).read_text())['data']:
            records.append((r['from'],r['to'],r['intensity']['actual']))
    c=pd.DataFrame(records,columns=['time','end','actual']);c['time']=pd.to_datetime(c.time,utc=True);c['end']=pd.to_datetime(c.end,utc=True)
    c=c.drop_duplicates('time').set_index('time').sort_index()
    if not ((c.end-c.index)==pd.Timedelta(minutes=30)).all():raise ValueError('碳数据时间粒度非法')
    h=pd.to_numeric(c.actual,errors='raise').resample('h').agg(['mean','count']);h.loc[h['count']!=2,'mean']=np.nan
    raw=root/'raw/opsd/time_series_60min.csv.gz'
    f=pd.read_csv(raw,usecols=['utc_timestamp',*COLUMNS.values()]).rename(columns={v:k for k,v in COLUMNS.items()})
    f.index=pd.to_datetime(f.pop('utc_timestamp'),utc=True)
    if f.index.duplicated().any():raise ValueError('重复时间戳')
    f=f.join(h['mean'].rename('carbon_g_per_kwh'));f['carbon_observed_g_per_kwh']=h['mean'].shift(1)
    f=f.loc['2019-04-01':'2019-06-30 23:00'];valid=[];dropped=[]
    for day,g in f.groupby(f.index.floor('D')):
        if len(g)!=24 or not np.isfinite(g.to_numpy()).all() or (g.drop(columns=['price'])<0).any().any():dropped.append(str(day.date()));continue
        g=g.copy();g['scenario']=str(day.date());g['step']=range(24);valid.append(g)
    if not valid:raise ValueError('没有完整日期')
    f=pd.concat(valid);old=json.loads((root/'gb/profiles/manifest.json').read_text())
    for k,v in old['scales'].items():f[k]*=v
    f['price']/=1000
    f[['scenario','step',*COLUMNS,'carbon_g_per_kwh','carbon_observed_g_per_kwh']].to_csv(out/'sealed.csv',index=False)
    m=dict(schema='phase12-confirmation-seal-v1',window=['2019-04-01','2019-06-30'],days=sorted(f.scenario.unique()),dropped=dropped,
           sha256=hashlib.sha256((out/'sealed.csv').read_bytes()).hexdigest(),scales=old['scales'],scaling_fit='2019-01 only',
           source_archive_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),carbon_sources=sources,
           usage='仅完整性检查；禁止第一、二阶段策略评价或调参。解封须独立锁定模型、偏好、种子和预算。',
           limits=['国家曲线缩放仿真，非馈线实测','碳观察为上一小时核算值代理，历史发布时间未核实','日间相关性；不假定日期独立'])
    (out/'seal.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='data/real');a=p.parse_args()
    m=seal(a.data_root);print(json.dumps({k:m[k] for k in ['window','days','dropped','sha256']},ensure_ascii=False,indent=2))
