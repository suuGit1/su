"""三条关键连接的行为验证，保留普通策略不改奖励口径。"""
import unittest
import numpy as np
from dataclasses import replace
from vpp_mappo.config import Config
from vpp_mappo.dispatch import DispatchSpec
from vpp_mappo.flex_resources import FlexSpec
from vpp_mappo.coordinator import TaskRiskCoordinator
from vpp_research.timing import CommandQueue,TimingContract
from vpp_research.safety import guard


class ConnectionTests(unittest.TestCase):
    def test_interval_guard_changes_action_and_rejects_impossible_box(self):
        class Network:
            def affine(self,row):return np.zeros((1,6)),[-1e6],[1e6]
        x=np.zeros(65);x[1]=.5;x[9]=.1
        f=FlexSpec(dr_shift_kw=0,dr_repay_kw=0,dr_backlog_kwh=0,dr_shift_budget_kwh=0,dr_shed_kw=0,dr_shed_budget_kwh=0)
        s=DispatchSpec();a=np.array([10000.,0,0,0,0,0])
        candidate,meta,certificate=guard(a,x,s,f,Network(),1,24)
        self.assertTrue(meta['guard_feasible']);self.assertTrue(certificate(candidate));self.assertFalse(certificate(a))
        x[54]=2.
        _,meta,certificate=guard(a,x,s,f,Network(),1,24)
        self.assertFalse(meta['guard_feasible']);self.assertFalse(certificate(candidate))
        self.assertIn(0,meta['guard_inconsistent_dimensions'])
    def test_task_and_uncertainty_change_actual_allocations(self):
        c=Config.load('configs/connected_ieee33.json')
        x=np.zeros(54);x[1]=.55
        agent=TaskRiskCoordinator(c,DispatchSpec(),FlexSpec())
        a,b,base=agent.allocate(x,np.ones(6),np.ones(6))
        agent.interval_width=np.array([.2,0,0,0,0,0,0,0,0])
        aa,bb,risk=agent.allocate(x,np.ones(6),np.ones(6))
        self.assertGreater(aa[0],a[0]);self.assertGreater(bb[0],b[0])
        self.assertGreater(risk['reserved_fraction'],base['reserved_fraction'])
        agent.interval_width=np.zeros(9);agent.config=replace(c,task_focus='carbon')
        ca,_,_=agent.allocate(x,np.ones(6),np.ones(6))
        self.assertGreater(ca[4],a[4])

    def test_nonzero_delivery_reaches_next_boundary_and_expired_is_removed(self):
        queue=CommandQueue(TimingContract(deadline_seconds=7200))
        queue.submit(0,np.ones(6),100,0)
        self.assertIsNone(queue.receive(0)[0])
        np.testing.assert_array_equal(queue.receive(3600)[0],np.ones(6))
        queue.submit(3600,np.ones(6),100,0)
        self.assertIsNone(queue.receive(12000)[0]);self.assertFalse(queue.pending)

if __name__=='__main__':unittest.main()
