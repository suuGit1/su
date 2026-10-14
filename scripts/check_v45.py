"""v4.5依赖、数据及代码包完整性检查；不执行训练、不打开确认集内容。"""
import os,sys
if __name__=='__main__' and not sys.flags.utf8_mode:
 os.environ['PYTHONUTF8']='1';os.execv(sys.executable,[sys.executable,'-X','utf8',*sys.argv])
import argparse,hashlib,importlib,importlib.metadata,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-bundle',action='store_true');a=p.parse_args()
 os.chdir(ROOT)
 versions={}
 for name in ('numpy','scipy','pandas','pandapower','torch','gymnasium','pettingzoo','matplotlib','openpyxl'):
  importlib.import_module(name);versions[name]=importlib.metadata.version(name)
 from vpp_mappo.config import Config
 from vpp_research.real_inputs import connect
 c,paths,sets,source=connect('data/real',Config.load('configs/v45_ieee33.json'))
 c.validate()
 subprocess.run([sys.executable,'validation/verify_vendor.py'],check=True)
 if a.verify_bundle:
  manifest=ROOT/'BUNDLE_MANIFEST.json'
  if not manifest.exists():raise ValueError('该目录不是已打包发行版；仓库检出请省略 --verify-bundle')
  records=json.loads(manifest.read_text(encoding='utf-8'))['files'];bad=[]
  for record in records:
   file=ROOT/record['path']
   if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:bad.append(record['path'])
  if bad:raise ValueError('文件缺失或摘要不同：'+', '.join(bad))
  print('代码包完整性校验通过：',len(records),'个文件；确认集只校验字节摘要、不用于指标计算')
 print(json.dumps(dict(version='4.5.0',python=sys.version.split()[0],dependencies=versions,days={k:len(v.profiles) for k,v in sets.items()},ev=source['ev_scope'],dr=source['dr']['mode'],network=c.network_model),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
