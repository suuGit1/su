"""审核可用真实数据与未满足的论文证据条件，不读取封存确认性数据。"""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.hardware_probe import probe
p=argparse.ArgumentParser();p.add_argument('--data-root',default='data/real');p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.data_root)
manifest=root/'ev/ornl_acn_sample/manifest.json';sample=json.loads(manifest.read_text()) if manifest.exists() else None
result=dict(catalog=json.loads((root/'catalog.json').read_text()),ev_sample=sample,hardware=probe(),
 blockers=['完整真实EV会话样本不足，不能据发现样本证明长期需求满足','GB真实DR参数缺失；SCE仅显式跨地区敏感性','边缘设备物理通信/能耗尚未实测'],
 provenance={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for folder in ('ev/ornl_acn_sample','dr/sce') for p in (root/folder).glob('*') if p.is_file()},
 clarification='catalog中的EV缺失是完整会话实验缺失；新增发现样本单列，未冒称完整数据')
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(ev_accepted=sample['accepted'] if sample else 0,blockers=result['blockers']),ensure_ascii=False))
