"""截取真实日曲线前两个小时，仅用于七方法集成验收，不用于论文性能比较。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    source=Path(source);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('验收数据目录非空')
    catalog=json.loads((source/'catalog.json').read_text(encoding='utf-8'))
    relative=catalog['default_profiles'];original=source/relative;target=output/relative
    target.mkdir(parents=True,exist_ok=True);hashes={}
    for split in ('train','validation','test'):
        path=original/(split+'.csv');hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8') as f,(target/path.name).open('w',newline='',encoding='utf-8') as out:
            reader=csv.DictReader(f);writer=csv.DictWriter(out,reader.fieldnames);writer.writeheader()
            writer.writerows(r for r in reader if int(r['step'])<2)
    manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'));manifest['horizon']=2
    manifest['acceptance_only']=True;manifest['parent_sha256']=hashes
    for split in ('train','validation','test'):
        manifest['splits'][split]['sha256']=hashlib.sha256((target/(split+'.csv')).read_bytes()).hexdigest()
    manifest['scope']='真实日曲线前两个小时，原始数值不改；仅验证接口、时序、审计和报告，不代表完整日调度'
    catalog['acceptance_only']=True
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='data/real');p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.source,a.output)
