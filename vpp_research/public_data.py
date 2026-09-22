"""英国 OPSD 与 NESO 碳强度的可追溯 UTC 对齐；禁止静默插值或跨地区冒充实测。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

COLUMNS={'load_kw':'GB_GBN_load_actual_entsoe_transparency','pv_kw':'GB_GBN_solar_generation_actual',
         'wind_kw':'GB_GBN_wind_generation_actual','price':'GB_GBN_price_day_ahead'}
BASE='https://api.carbonintensity.org.uk/intensity/date/'


def download_carbon(folder,start='2018-12-31',end='2019-04-01'):
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    dates=[];d=date.fromisoformat(start)
    while d<date.fromisoformat(end):dates.append(d.isoformat());d+=timedelta(days=1)
    def get(day):
        p=out/(day+'.json')
        if not p.exists():
            with urllib.request.urlopen(BASE+day,timeout=30) as r:raw=r.read()
            value=json.loads(raw)
            if not isinstance(value.get('data'),list) or not value['data']:raise ValueError(day+' 无碳数据')
            temp=p.with_suffix('.part');temp.write_bytes(raw);temp.replace(p)
        return dict(date=day,url=BASE+day,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    # 少量并发下载互不依赖的公开日文件；任何失败使整个准备流程中止。
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(get,dates))
    (out/'manifest.json').write_text(json.dumps(records,indent=2));return records


def prepare(raw_opsd,carbon_folder,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('输出目录非空')
    carbon=[]
    for p in sorted(Path(carbon_folder).glob('????-??-??.json')):
        for r in json.loads(p.read_text())['data']:
            carbon.append(dict(timestamp=pd.Timestamp(r['from']),end=pd.Timestamp(r['to']),actual=r['intensity']['actual']))
    c=pd.DataFrame(carbon).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    if not (c.end-c.index==pd.Timedelta(minutes=30)).all():raise ValueError('非半小时碳区间')
    actual=pd.to_numeric(c.actual,errors='raise');hour=actual.resample('h').agg(['mean','count'])
    hour.loc[hour['count']!=2,'mean']=np.nan
    df=pd.read_csv(raw_opsd,usecols=['utc_timestamp',*COLUMNS.values()]).rename(columns={v:k for k,v in COLUMNS.items()})
    df['timestamp']=pd.to_datetime(df.pop('utc_timestamp'),utc=True)
    if df.timestamp.duplicated().any():raise ValueError('OPSD UTC 时间重复')
    df=df.set_index('timestamp').join(hour['mean'].rename('carbon_g_per_kwh'))
    # 使用上一小时结束的核算值作为滞后观测代理；发布延迟尚无原始档案，须显式披露。
    df['carbon_observed_g_per_kwh']=hour['mean'].shift(1)
    df=df.loc['2019-01-01':'2019-03-31 23:00:00'];valid=[];dropped=[]
    for day,g in df.groupby(df.index.floor('D')):
        values=g.to_numpy(dtype=float)
        if len(g)!=24 or not np.isfinite(values).all() or (g[['load_kw','pv_kw','wind_kw','carbon_g_per_kwh','carbon_observed_g_per_kwh']]<0).any().any():
            dropped.append(str(day.date()));continue
        g=g.copy();g['scenario']=str(day.date());g['step']=range(24);valid.append(g)
    if not valid:raise ValueError('没有时空完整匹配的日期')
    df=pd.concat(valid);train=df[df.index<'2019-02-01'];scale={k:v/float(train[k].max()) for k,v in [('load_kw',1800),('pv_kw',700),('wind_kw',300)]}
    if not all(np.isfinite(v) and v>0 for v in scale.values()):raise ValueError('训练缩放不可用')
    for k,v in scale.items():df[k]*=v
    df['price']/=1000;out.mkdir(parents=True)
    m=dict(sources=['https://data.open-power-system-data.org/time_series/2020-10-06/',BASE],region='GB',timezone='UTC',currency='GBP',price_conversion='GBP/MWh 除以 1000 得 GBP/kWh；退化、DR 和通信费用参数按 GBP 假设声明',
        horizon=24,dt_hours=1.,scales=scale,scaling_fit='2019-01 only',dropped_days=dropped,
        raw_opsd_sha256=hashlib.sha256(Path(raw_opsd).read_bytes()).hexdigest(),
        carbon_sources=json.loads((Path(carbon_folder)/'manifest.json').read_text()),
        carbon_scope='NESO 电力运行 CO2 估计；不包含设备生命周期；上小时实际值作为滞后可用性假设，非已核验历史发布版本',
        scope='英国国家曲线缩放至 IEEE33；真实曲线驱动仿真，不是同一馈线实测',splits={})
    for name,left,right in [('train','2019-01-01','2019-02-01'),('validation','2019-02-01','2019-03-01'),('test','2019-03-01','2019-04-01')]:
        part=df[(df.index>=left)&(df.index<right)]
        if part.empty:raise ValueError(name+' 缺少有效数据')
        p=out/(name+'.csv');part[['scenario','step',*COLUMNS,'carbon_g_per_kwh','carbon_observed_g_per_kwh']].to_csv(p,index=False)
        m['splits'][name]=dict(days=part.scenario.nunique(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--opsd',required=True);p.add_argument('--carbon-folder',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    download_carbon(a.carbon_folder);m=prepare(a.opsd,a.carbon_folder,a.output);print(json.dumps(m['splits'],indent=2))
