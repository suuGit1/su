"""将已规范化的会话 CSV 转成按场景索引的 JSON；不推断未知需求。"""
import argparse
import csv
import json
import math
from pathlib import Path
from .flex_resources import validate_sessions


def convert(source,output,horizon,dt_hours,provenance):
    if horizon<1 or not math.isfinite(dt_hours) or dt_hours<=0 or not provenance.strip():raise ValueError('时域、步长和来源不能为空或非法')
    target=Path(output)
    if target.exists():raise ValueError('输出文件已存在')
    groups={}
    with Path(source).open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or [])!={'scenario','id','arrival_step','departure_step','energy_kwh','max_kw'}:raise ValueError('CSV 字段与会话规范不一致')
        for row in reader:
            key=row.pop('scenario')
            if not key:raise ValueError('场景名不能为空')
            for k in ('arrival_step','departure_step'):row[k]=int(row[k])
            for k in ('energy_kwh','max_kw'):row[k]=float(row[k])
            groups.setdefault(key,[]).append(row)
    if not groups:raise ValueError('CSV 没有会话；零车辆场景请在 JSON 中显式填空数组')
    groups={k:validate_sessions(v,horizon,dt_hours) for k,v in groups.items()}
    result=dict(schema_version=1,energy_basis='grid_kwh',source=provenance,horizon=horizon,dt_hours=dt_hours,scenarios=groups)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--horizon',type=int,required=True);p.add_argument('--dt-hours',type=float,required=True)
    p.add_argument('--source',required=True,help='数据出处及声明需求/期限的处理方法')
    a=p.parse_args();convert(a.csv,a.output,a.horizon,a.dt_hours,a.source)

if __name__=='__main__':main()
