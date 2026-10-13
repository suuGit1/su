"""公共终端协议的符号、鲁棒边界、MILP与后备语义回归。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.dispatch import DispatchSpec
from vpp_mappo.terminal_service import band,reachable_band,power_bounds,shortfall_kwh
from vpp_mappo.flex_resources import FlexSpec
from vpp_mappo.flex_optimization import solve_flex

class EmptyNetwork:
    def affine(self,row):return np.zeros((0,6)),np.zeros(0),np.zeros(0)

class TerminalTests(unittest.TestCase):
    def test_inverse_and_reachability(self):
        s=DispatchSpec();target=(.54,.56)
        for remaining in (0,1,3):
            interval=reachable_band(s,target,remaining,1.)
            for soc in (.1,.3,.55,.9):
                lo,hi=power_bounds(s,soc,soc,interval,1.)
                if lo<=hi:
                    for a in np.linspace(lo,hi,9):
                        after=s.next_soc([soc,0],[a,0],1.)[0]
                        self.assertTrue(interval[0]-1e-10<=after<=interval[1]+1e-10)
    def test_interval_conflict(self):
        lo,hi=power_bounds(DispatchSpec(),.4,.6,(.54,.56),1.)
        self.assertGreater(lo,hi)
    def test_all_objectives_terminal(self):
        s=DispatchSpec();f=FlexSpec();rows=[dict(load_kw=1000,pv_kw=0,wind_kw=0,price=.2)]*3
        for mode in ('economic','min_grid','max_grid'):
            plans,m=solve_flex(s,f,EmptyNetwork(),rows,0,.2,0,0,0,[],{},1.,1000,objective=mode,terminal_band=(.54,.56))
            soc=np.array([.2,0.])
            for p in plans:soc=s.next_soc(soc,[p['action'][0],0],1.)
            self.assertLess(shortfall_kwh(soc[0],(.54,.56),500),1e-5)
    def test_legacy_default_and_validation(self):
        self.assertIsNone(band(Config(),DispatchSpec()))
        with self.assertRaises(ValueError):Config(terminal_soc_mode='shared_band').validate()
    def test_feasible_rejects_service_failure(self):
        from vpp_research.suite import feasible
        row=dict(failed=False,violations=0,ac_violations=0,ac_failed=0,reserve_invalid=0,ev_unmet_kwh=0,dr_backlog_kwh=0,terminal_service_failed=True)
        self.assertFalse(feasible(row))

if __name__=='__main__':unittest.main()
