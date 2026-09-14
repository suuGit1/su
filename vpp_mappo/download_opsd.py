"""下载并校验固定版本 OPSD 原始数据，避免把不完整下载交给预处理。"""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL='https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
SHA256='6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8'


def download(output):
    path=Path(output)
    if path.exists(): raise ValueError('目标文件已存在，请直接预处理或使用新路径')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    if temp.exists(): raise ValueError('存在未完成的 .part 文件，请检查后移走再试')
    digest=hashlib.sha256(); size=0
    try:
        with urllib.request.urlopen(URL,timeout=60) as r,temp.open('xb') as f:
            while True:
                block=r.read(1024*1024)
                if not block: break
                f.write(block); digest.update(block); size+=len(block)
                if size%(16*1024*1024)==0: print(f'已下载 {size//1024//1024} MiB',flush=True)
        if digest.hexdigest()!=SHA256: raise ValueError('源文件哈希与已核验版本不一致')
        temp.replace(path)
    except Exception:
        if temp.exists(): temp.unlink()
        raise
    print(f'下载并校验完成：{path}，{size} 字节')


if __name__=='__main__':
    p=argparse.ArgumentParser(description='下载 OPSD 2020-10-06 小时数据，约 124 MiB')
    p.add_argument('--output',default='data/raw/time_series_60min_singleindex.csv')
    download(p.parse_args().output)
