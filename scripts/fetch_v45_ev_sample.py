"""获取ORNL公开26条ACN发现样本，保留因果申报字段；不当作完整站点数据。"""
import os as _os, sys as _sys
if __name__ == '__main__' and not _sys.flags.utf8_mode:
    _os.environ['PYTHONUTF8']='1'
    _os.execv(_sys.executable,[_sys.executable,'-X','utf8',*_sys.argv])

import sys,json,hashlib
from pathlib import Path
from urllib.request import urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.ev_import import convert
URL='https://ornl.opendatasoft.com/api/explore/v2.1/catalog/datasets/acn-data/records?limit=100'
root=Path('data/real/ev/ornl_acn_sample');root.mkdir(parents=True,exist_ok=True)
raw=root/'source.json'
if not raw.exists():
 with urlopen(URL,timeout=30) as response:raw.write_bytes(response.read())
data=json.loads(raw.read_text());items=[]
for r in data['results']:
 items.append(dict(connectionTime=r['connectiontime'],disconnectTime=r['disconnecttime'],sessionID=r['sessionid'],userInputs=json.loads(r['userinputs']) if r.get('userinputs') else []))
normalized=root/'normalized.json';normalized.write_text(json.dumps(items,ensure_ascii=False,indent=2))
bundle=root/'sessions.json'
if not bundle.exists():audit=convert(normalized,bundle,dt_hours=1.,max_kw=7.,source=URL,replay_actual_departures=True)
else:audit=json.loads(bundle.with_suffix('.audit.json').read_text())
manifest=dict(source=URL,source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),input_count=len(items),accepted=audit['accepted'],sample_only=True,complete_site_dataset=False,
 scope='ORNL仅采集26条用于数据发现，不能证明完整EV需求或长期效果；美国2020年会话，不与GB2019年默认为同地区同日',assumed_max_kw=7.,causal_import_audit='sessions.audit.json')
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False))
