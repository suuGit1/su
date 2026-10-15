"""生成含源码、已取得真实数据和开发示例的独立v4.5发行包。"""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build(output):
 output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
 selected=set()
 for folder in ('vpp_mappo','vpp_research','vendor','src','configs','scripts','validation','tests','docs','examples/v45','data/real','release_evidence/v45'):
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not any(x.startswith('.') for x in p.relative_to(ROOT).parts) and p.suffix not in ('.pyc','.tmp'):selected.add(p)
 for pattern in ('run*.py','requirements*.txt','README*.md','LICENSE','VERSION','pyproject.toml'):
  selected.update(ROOT.glob(pattern))
 selected.add(ROOT/'experiments/v42/dt_study_evidence.zip')
 # 程序能读取训练、校准和测试所需数据；确认集只复制原字节，不计算其指标。
 records=[]
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(selected):
   data=p.read_bytes();relative=p.relative_to(ROOT).as_posix()
   z.writestr('dt-c3-pareto-vpp-v4.5.0/'+relative,data)
   records.append(dict(path=relative,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
  m=dict(version='4.5.0',release_date='2026-10-14',files=records,scope='可运行研究代码＋已取得公开数据＋开发摘要；不含依赖二进制和历史训练中间状态')
  z.writestr('dt-c3-pareto-vpp-v4.5.0/BUNDLE_MANIFEST.json',json.dumps(m,ensure_ascii=False,indent=2))
 with zipfile.ZipFile(output) as z:
  bad=z.testzip()
  if bad:raise ValueError('打包CRC校验失败：'+bad)
 print(json.dumps(dict(path=str(output),files=len(records)+1,bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest()),ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);build(p.parse_args().output)
