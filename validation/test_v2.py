"""v2针对虚假DR外推和备用时刻错位的回归检查。"""
import unittest
from dataclasses import asdict
import numpy as np
from vpp_mappo.basic_dt import BasicDT
from vpp_mappo.dispatch import DispatchSpec
from vpp_mappo.flex_resources import FlexSpec
from vpp_mappo.config import Config
from vpp_research.environment import ResearchEnv
from vpp_research.safety import guard


class V2Tests(unittest.TestCase):
    def test_disabled_dr_cannot_grow_from_unexecuted_candidate(self):
        f=FlexSpec(dr_shift_kw=0,dr_repay_kw=0,dr_backlog_kwh=0,dr_shift_budget_kwh=0,dr_shed_kw=0,dr_shed_budget_kwh=0)
        payload=[{'soc':.5},{'sessions':[]},{'backlog':0,'shifted':0,'load_kw':1000,'price':.1},
                 {'shed_used':0},{'pv_kw':0},{'wind_kw':0}]
        dt=BasicDT(DispatchSpec(),f,1,payload)
        for _ in range(10):dt.command([0,0,999,999,0,0])
        predicted=dt.estimate(10)
        self.assertEqual(predicted[2]['shifted'],0);self.assertEqual(predicted[3]['shed_used'],0)

    def test_reserve_audits_same_period_including_last_step(self):
        c=Config.load('configs/research_smoke.json');c.horizon=2
        e=ResearchEnv(c,reserve_mode='pcc_checked');e.reset(91)
        for step in range(2):
            *_,info=e.step(np.zeros((e.num_agents,1)))
            proof=info['reserve_confirmation']
            self.assertEqual(proof['state_step'],step);self.assertEqual(proof['execution_step'],step)
            self.assertTrue(proof['baseline_feasible']);self.assertTrue(info['reserve_valid'])
            self.assertEqual(info['objective_version'],'c3-ac-pcc-same-period-sampled-reserve-v4')

    def test_large_soc_interval_intersects_known_physical_support(self):
        class Network:
            def affine(self,row):return np.zeros((1,6)),[-1e6],[1e6]
        x=np.zeros(65);x[1]=.5;x[54]=2.
        f=FlexSpec(dr_shift_kw=0,dr_repay_kw=0,dr_backlog_kwh=0,dr_shift_budget_kwh=0,dr_shed_kw=0,dr_shed_budget_kwh=0)
        a,m,cert=guard(np.zeros(6),x,DispatchSpec(),f,Network(),1,24)
        self.assertTrue(m['guard_feasible']);self.assertTrue(cert(a));self.assertAlmostEqual(a[0],0)
