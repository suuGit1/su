"""真实EV发现样本与SCE参数接入验收；显式跨地域组合，不冒充同站点实测。"""
import argparse,copy,csv,json,sys,zipfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_mappo.config import Config
from vpp_research.real_inputs import connect
from vpp_research.integrated import rollout
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
c,paths,sets,source=connect('data/real',Config.load('configs/v45_ieee33.json'),dr_mode='sce-derated')
bundle_path=Path('data/real/ev/ornl_acn_sample/sessions.json');bundle=json.loads(bundle_path.read_text());dates=sorted(bundle['scenarios'])
if len(dates)!=1:raise ValueError('当前验收协议要求已审核的单日发现样本')
date=dates[0];energy_date='2019-02-27';csv_path=out/'composed.csv'
with paths['validation'].open() as src,csv_path.open('w',newline='') as dst:
 r=csv.DictReader(src);w=csv.DictWriter(dst,r.fieldnames);w.writeheader()
 for row in r:
  if row['scenario']==energy_date:row['scenario']=date;w.writerow(row)
c._ev_bundle=bundle;c.coordinator_active_channels=(1,1,1,1,1,1)
with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:model=json.loads(z.read('physics.json'))
manifest=dict(energy_original_date=energy_date,ev_original_date=date,source=source,ev_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
 scope='GB2019能源曲线+美国2020真实EV发现样本+SCE2022DR参数的显式组合，仅接口与约束验收；不是同站点或同日真实VPP',
 dt_scope='转移的物理DT区间未针对EV/DR重新校准，不宣称覆盖率保证',synthetic_ev=False)
(out/'composition.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
r=rollout(c,model,out/'mpc',method='mpc',seed=83000,csv_path=csv_path,ev_bundle=bundle)
print(json.dumps(r,ensure_ascii=False))
