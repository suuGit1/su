"""在真实训练日期相同PCC目标下比较经济求解和纯可行性；只核验可执行性。"""
import sys,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vpp_mappo.config import Config
from vpp_research.real_inputs import connect
from vpp_research.ac_safety import ACCheckedFlex
from vpp_research.train_v4 import training_context
from vpp_research.pcc_reserve import target_plan
c,paths,sets,source=connect('data/real',Config.load('configs/v45_ieee33.json'));c.seed=43;c.solver_time_limit=2.
rows=[]
for ep in (10,15):
 day,seed=training_context(c,ep,len(sets['train'].profiles))
 for flag in (False,True):
  cfg=copy.deepcopy(c);cfg.reserve_feasibility_only=flag;core=ACCheckedFlex(cfg,paths['train']);core.reset(seed,day)
  center=core.network.audit(core.row(),np.zeros(6))['ac_grid_kw']
  for offset in (-100.,100.):
   t=time.perf_counter();r=dict(episode=ep,day=day,feasibility_only=flag,target=center+offset,failed=False)
   try:
    plan,m=target_plan(core,center+offset);r.update(error=m['pcc_error_kw'],physical_violations=core.check(core.row(),plan['action'],plan['ev_kw']),ac_violations=core.network.audit(core.row(),plan['action'])['ac_violations'])
   except Exception as exc:r.update(failed=True,error=str(exc))
   r['seconds']=time.perf_counter()-t;rows.append(r);print(json.dumps(r),flush=True)
Path('experiments/v45/reserve_speed_probe.json').write_text(json.dumps(rows,indent=2))
