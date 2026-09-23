"""从统一目录的原始文件重建真实数据；不生成或补齐合成曲线。"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.public_data import prepare
from vpp_research.dr_import import convert
from vpp_research.real_inputs import connect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/real');p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output',help='重建到新的目录，不覆盖现有切分')
    a=p.parse_args();root=Path(a.data_root)
    if a.verify_only:
        _,_,sets,protocol=connect(root)
        print(json.dumps(dict(days={k:len(d.profiles) for k,d in sets.items()},protocol=protocol),ensure_ascii=False,indent=2));return
    if not a.output:p.error('重建要求 --output 新目录；只检查请用 --verify-only')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('输出目录非空')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'catalog.json',out/'catalog.json')
    with tempfile.TemporaryDirectory() as temp:
        raw=Path(temp)/'opsd.csv'
        with gzip.open(root/'raw/opsd/time_series_60min.csv.gz','rb') as src,raw.open('wb') as dest:shutil.copyfileobj(src,dest)
        from vpp_mappo.download_opsd import SHA256
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=SHA256:
            raise ValueError('OPSD原始文件不完整或版本不匹配；请使用download_opsd重新获取固定版本')
        prepare(raw,root/'raw/neso',out/'gb/profiles')
    convert(root/'dr/sce/source.xlsx',out/'dr/sce/events.json')
    shutil.copytree(root/'raw',out/'raw')
    shutil.copyfile(root/'dr/sce/source.xlsx',out/'dr/sce/source.xlsx')
    if (root/'archive').exists():shutil.copytree(root/'archive',out/'archive')
    print('已重建真实GB日曲线与SCE事件：',out.resolve())


if __name__=='__main__':main()
