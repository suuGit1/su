"""任务风险对可行服务区间的剩余额度必须有效，同时保留actor控制权。"""
import os as _os, sys as _sys
if __name__ == '__main__' and not _sys.flags.utf8_mode:
    _os.environ['PYTHONUTF8']='1'
    _os.execv(_sys.executable,[_sys.executable,'-X','utf8',*_sys.argv])

import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from types import SimpleNamespace
from vpp_mappo.config import Config
from vpp_mappo.cyber import CyberSpec
from vpp_mappo.coordinator_projection import PublicServiceProjection
class Tests(unittest.TestCase):
 def test_task_and_floor_and_actor(self):
  c=Config(coordinator_priority_slack_fraction=.25,coordinator_mode='schedule',risk_adaptive=True,dt_hours=1,coordinator_active_channels=(1,0,1,0,1,1))
  q=PublicServiceProjection(c,SimpleNamespace(soc_min=[.1],soc_max=[.9]),SimpleNamespace(ev_station_kw=100,dr_backlog_kwh=0,dr_repay_kw=0),CyberSpec())
  x=np.zeros(54);x[1]=.5;q.interval_width=np.zeros(9)
  c.task_focus='economic';a,b,r=q.allocate(x,np.ones(6),np.ones(6))
  c.task_focus='reserve';d,e,s=q.allocate(x,np.ones(6),np.ones(6))
  self.assertTrue(r['projection_applied']);self.assertGreater(np.linalg.norm(a-d),1e-5)
  self.assertTrue(np.all(a>=np.array(r['bandwidth_floor'])-1e-12));self.assertAlmostEqual(a.sum(),1.)
  f,g,_=q.allocate(x,[9,0,1,0,1,1],np.ones(6));self.assertGreater(np.linalg.norm(d-f),1e-5);np.testing.assert_allclose(e,g)
 def test_off_preserves_legacy(self):
  c=Config();self.assertEqual(c.coordinator_priority_slack_fraction,0.)
  with self.assertRaises(ValueError):Config(coordinator_priority_slack_fraction=2).validate()
if __name__=='__main__':unittest.main()
