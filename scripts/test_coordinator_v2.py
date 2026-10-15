"""检查会影响安全和资源公平性的协调器接口契约。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.coordinator_v2 import MaskedRiskCoordinator

class Contracts(unittest.TestCase):
    def make(self):
        c=Config(coordinator_mode='schedule',risk_adaptive=True)
        return MaskedRiskCoordinator(c,SimpleNamespace(soc_min=[.1],soc_max=[.9]),SimpleNamespace(ev_station_kw=100,dr_backlog_kwh=100,dr_repay_kw=50),[1,0,1,0,1,1])
    def test_resource_mask(self):
        q=self.make();x=np.zeros(54);x[1]=.5
        for v in (np.zeros(6),np.ones(6),np.arange(6)):
            bw,cpu,r=q.allocate(x,v,v)
            for a in (bw,cpu):
                self.assertAlmostEqual(a.sum(),1);self.assertTrue((a>=0).all());self.assertEqual(a[1]+a[3],0)
    def test_load_and_price_risk(self):
        q=self.make();x=np.zeros(54);x[1]=.5
        for j in (2,3,5,6):
            q.interval_width=np.zeros(9);q.interval_width[j]=.1;r=q.assess(x)
            self.assertEqual(r['interval_risk'][2],2.)
    def test_reject_invalid(self):
        q=self.make();x=np.zeros(54)
        with self.assertRaises(ValueError):q.allocate(x,[-1]*6,[1]*6)
        q.interval_width=np.full(9,np.nan)
        with self.assertRaises(ValueError):q.assess(x)
if __name__=='__main__':unittest.main()
