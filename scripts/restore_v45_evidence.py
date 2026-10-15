"""校验并合并分卷证据包；不覆盖内容不同的已有文件。"""
import os as _os, sys as _sys
if __name__ == '__main__' and not _sys.flags.utf8_mode:
    _os.environ['PYTHONUTF8']='1'
    _os.execv(_sys.executable,[_sys.executable,'-X','utf8',*_sys.argv])

import argparse,hashlib,json,os
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def restore(manifest,output=None):
 manifest=Path(manifest);m=json.loads(manifest.read_text());out=Path(output) if output else manifest.parent/m['output']
 if out.exists():
  if out.stat().st_size!=m['size'] or digest(out)!=m['sha256']:raise ValueError('目标已存在且摘要不同，禁止覆盖')
  return out
 temp=out.with_suffix(out.suffix+'.assembling')
 with temp.open('xb') as dst:
  for item in m['parts']:
   if Path(item['path']).name!=item['path']:raise ValueError('分卷必须与清单同目录')
   part=manifest.parent/item['path']
   if part.stat().st_size!=item['size'] or digest(part)!=item['sha256']:raise ValueError('分卷损坏：'+str(part))
   with part.open('rb') as src:
    for chunk in iter(lambda:src.read(1024*1024),b''):dst.write(chunk)
  dst.flush();os.fsync(dst.fileno())
 if temp.stat().st_size!=m['size'] or digest(temp)!=m['sha256']:raise ValueError('合并包摘要不匹配')
 temp.replace(out);return out

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--output');a=p.parse_args();print(restore(a.manifest,a.output))
