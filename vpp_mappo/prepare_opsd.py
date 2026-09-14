"""真实 OPSD 小时数据：按 UTC 整日清洗、时间切分、仅训练集拟合缩放。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = 'https://data.open-power-system-data.org/time_series/2020-10-06/'
COLUMNS = {'load_kw':'DE_load_actual_entsoe_transparency','pv_kw':'DE_solar_generation_actual',
           'wind_kw':'DE_wind_generation_actual','price':'DE_LU_price_day_ahead'}


def prepare(raw, output, start='2019-01-01', train_end='2019-09-01', validation_end='2019-11-01', end='2020-01-01', load_peak=1800., pv_peak=700., wind_peak=300.):
    cuts = [pd.Timestamp(x,tz='UTC') for x in (start,train_end,validation_end,end)]
    if not all(a < b for a,b in zip(cuts,cuts[1:])): raise ValueError('时间边界必须严格递增')
    if not np.isfinite([load_peak,pv_peak,wind_peak]).all() or min(load_peak,pv_peak,wind_peak)<=0:
        raise ValueError('缩放目标必须为有限正数')
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('输出目录非空')
    df = pd.read_csv(raw,usecols=['utc_timestamp',*COLUMNS.values()])
    df['timestamp'] = pd.to_datetime(df.pop('utc_timestamp'),utc=True,errors='raise')
    df = df[(df.timestamp>=cuts[0])&(df.timestamp<cuts[-1])].sort_values('timestamp')
    if df.timestamp.duplicated().any(): raise ValueError('存在重复 UTC 时间戳')
    df = df.rename(columns={v:k for k,v in COLUMNS.items()})
    valid, dropped = [], []
    for day,g in df.groupby(df.timestamp.dt.floor('D')):
        expected = pd.date_range(day,periods=24,freq='h')
        numeric = g[list(COLUMNS)].to_numpy(dtype=float)
        if len(g)!=24 or not np.array_equal(g.timestamp.to_numpy(),expected.to_numpy()) or not np.isfinite(numeric).all() or (numeric[:,:3]<0).any():
            dropped.append(str(day.date())); continue
        g = g.copy(); g['scenario'] = str(day.date()); g['step'] = range(24); valid.append(g)
    if not valid: raise ValueError('没有完整有效的 24 小时场景')
    clean = pd.concat(valid,ignore_index=True)
    train = clean[clean.timestamp<cuts[1]]
    denom = train[['load_kw','pv_kw','wind_kw']].max()
    if len(train)==0 or (denom<=0).any(): raise ValueError('训练段缺失或出力全零，无法拟合缩放')
    scales = dict(zip(('load_kw','pv_kw','wind_kw'),np.array([load_peak,pv_peak,wind_peak])/denom.to_numpy()))
    for k,v in scales.items(): clean[k] *= v
    # 上游功率为 MW，按训练峰值映射到研究馈线 kW；EUR/MWh 换算为 EUR/kWh。
    clean['price'] /= 1000
    out.mkdir(parents=True,exist_ok=True)
    digest = hashlib.sha256()
    with open(raw,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    manifest = dict(source=SOURCE, raw_file_sha256=digest.hexdigest(),
        original_columns=COLUMNS, timezone='UTC',horizon=24,dt_hours=1.0,
        power_mapping='国家级实测曲线按训练峰值缩放，非原生馈线实测',currency='EUR',
        scales=scales, fitted_on='train only',boundaries=[str(x) for x in cuts],dropped_days=dropped,splits={})
    for name,left,right in zip(('train','validation','test'),cuts[:-1],cuts[1:]):
        part = clean[(clean.timestamp>=left)&(clean.timestamp<right)]
        if part.empty: raise ValueError(name+' 没有有效场景')
        path = out/(name+'.csv')
        part[['scenario','step',*COLUMNS]].to_csv(path,index=False)
        manifest['splits'][name] = dict(days=int(part.scenario.nunique()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest['splits'],ensure_ascii=False,indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description='转换 OPSD 2020-10-06 的 60min singleindex 数据')
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    for key,default in [('start','2019-01-01'),('train-end','2019-09-01'),('validation-end','2019-11-01'),('end','2020-01-01')]: p.add_argument('--'+key,default=default)
    p.add_argument('--load-peak',type=float,default=1800); p.add_argument('--pv-peak',type=float,default=700); p.add_argument('--wind-peak',type=float,default=300)
    a = p.parse_args(); prepare(a.input,a.output,a.start,a.train_end,a.validation_end,a.end,a.load_peak,a.pv_peak,a.wind_peak)


if __name__ == '__main__': main()
