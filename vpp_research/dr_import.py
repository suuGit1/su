"""CPUC/SCE 2022 A.4 VPP 真实 DR 事件导入；保留负响应和事后评价口径。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE='https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/energy-division/documents/demand-response/emergency-load-reduction-program/elrp-2022-program-data/sce-elrp-hourly-with-lip-values.xlsx'


def convert(source, output):
    path=Path(source)
    nominations=pd.read_excel(path,sheet_name='Nominations',header=None)
    hours=pd.read_excel(path,sheet_name='Hourly Performance',header=None)
    if 'MWh' not in str(hours.iloc[0,8]) or nominations.iloc[1,14]!='A.4' or hours.iloc[1,36]!='A.4 VPP' or hours.iloc[2,36]!='Net':
        raise ValueError('SCE 工作簿结构改变；禁止按旧列号静默读取')
    nominated={}
    for _,r in nominations.iloc[2:].iterrows():
        if isinstance(r.iloc[0],pd.Timestamp) or hasattr(r.iloc[0],'date'):
            day=r.iloc[0].date().isoformat()
            if day in nominated:raise ValueError('重复提名日期')
            nominated[day]=float(r.iloc[14]) if pd.notna(r.iloc[14]) else None
    rows=[];keys=set()
    for _,r in hours.iloc[3:].iterrows():
        if not hasattr(r.iloc[0],'date'):continue
        day=r.iloc[0].date().isoformat();hour=int(r.iloc[1]);key=(day,hour)
        if key in keys:raise ValueError('重复事件小时')
        keys.add(key)
        if not 1<=hour<=24:raise ValueError('Hour Ending 超出1–24')
        duration=float(r.iloc[6])
        if not np.isfinite(duration) or not 0<=duration<=1:raise ValueError('小时内事件时长非法')
        nomination=nominated.get(day)
        def number(i):return float(r.iloc[i]) if pd.notna(r.iloc[i]) and np.isfinite(float(r.iloc[i])) else None
        net,adjusted,aggregate,lip=[number(i) for i in (36,37,38,39)]
        eligible=duration>0 and nomination is not None and nomination>0 and net is not None
        rows.append(dict(event_date=day,hour_ending_local=hour,
            timezone_assumption='America/Los_Angeles；原表未标时区，未擅自与UTC曲线拼接',
            duration_hours=duration,nominated_mw=nomination,net_mwh=net,adjusted_mwh=adjusted,aggregate_mwh=aggregate,lip_mwh=lip,
            response_ratio=net/(nomination*duration) if eligible else None,
            eligible=eligible,split='train' if day<='2022-09-04' else 'calibration' if day<='2022-09-06' else 'test'))
    train=np.array([r['response_ratio'] for r in rows if r['eligible'] and r['split']=='train'])
    if len(train)==0:raise ValueError('没有可用于历史参数估计的训练事件')
    result=dict(schema_version='sce-a4-dr-events-v1',source=SOURCE,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        subgroup='SCE ELRP A.4 VPP',measurement='事后基线法 Net incremental load reduction；不是直接测得的可控设备容量',
        rows=rows,summary=dict(hourly_rows=len(rows),active_rows=sum(r['eligible'] for r in rows),
            negative_response_rows=sum(r['eligible'] and r['net_mwh']<0 for r in rows),
            train_ratio_q10=float(np.quantile(train,.1)),train_ratio_median=float(np.median(train))),
        limits=['MW×实际事件时长才可与MWh相除；非事件小时不算零响应',
            '负值和大于1的响应比例完整保留，不冒称物理控制系数',
            '仅训练日期用于参数估计；测试表现不能反向定义设备约束',
            '加州项目与GB曲线属于跨来源仿真，不能宣称同一实站',
            '该数据不能识别DR回补动态、单设备爬坡或真实成本参数'])
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result


def conservative_flex(base_record, events):
    """把历史训练分位数转成显式降额情景；不是有统计保证的可调容量估计。"""
    q=float(events['summary']['train_ratio_q10'])
    if not np.isfinite(q):raise ValueError('训练分位数非法')
    factor=max(0.,min(1.,q));record=dict(base_record)
    record['dr_shed_kw']*=factor;record['dr_shed_budget_kwh']*=factor
    return record,dict(derating_factor=factor,certified=False,
        source=events['source'],description='仅削减DR降额敏感性参数；平移DR、回补及成本仍为原研究假设')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=convert(a.input,a.output);print(result['summary'])
