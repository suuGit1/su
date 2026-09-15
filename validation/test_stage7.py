"""观察等价 MPC、规则协调器及共享执行器验证。"""
import unittest
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.environment import VPPAdapter
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_mappo.coordinator import TaskRiskCoordinator


class Stage7Tests(unittest.TestCase):
    def env(self):
        c=Config.load('configs/c3_coordinated_smoke.json');e=VPPAdapter(c);e.reset(1);return e
    def test_mpc_cannot_see_unreceived_truth(self):
        a=self.env();b=self.env();b.core.soc[0]=.8;b.core.remaining['demo_0']=.1
        b.core.profiles['wind_kw'][1:]+=100;b.core.sessions[1]['departure_step']=7
        x=a.encode()[0][0];y=b.encode()[0][0];np.testing.assert_array_equal(x,y)
        p=ObservedMPC(a.config,a.spec,a.flex,2)
        aa,ma=p.propose(x);bb,mb=p.propose(y)
        self.assertFalse(ma['mpc_failed']);np.testing.assert_allclose(aa,bb)
        self.assertEqual(ma['mpc_ev_model'],mb['mpc_ev_model'])
    def test_mpc_failure_explicit(self):
        e=self.env();x=e.encode()[0][0].copy();x[1]=2
        a,m=ObservedMPC(e.config,e.spec,e.flex).propose(x)
        self.assertTrue(m['mpc_failed']);self.assertTrue(m['mpc_failure_reason']);np.testing.assert_array_equal(a,np.zeros(6))
        self.assertGreater(m['mpc_solver_seconds'],0)
    def test_coordinator_priority_and_budget(self):
        e=self.env();x=e.encode()[0][0].copy();x[18:24]=0;x[19]=10
        bw,cpu,r=e.coordinator.allocate(x,np.ones(6),np.ones(6))
        self.assertGreater(bw[1],bw[4]);self.assertLessEqual(bw.sum(),1+1e-12);self.assertLessEqual(cpu.sum(),1+1e-12)
        self.assertIn('stale_telemetry',r['tasks'][1]['reasons'])
    def test_task_deadline_pressure(self):
        e=self.env();x=e.encode()[0][0].copy();x[3]=.3;x[5]=1/e.config.horizon
        r=e.coordinator.assess(x)
        self.assertTrue(any(t['kind']=='ev_service' for t in r['tasks']))
        self.assertIn('ev_deadline_pressure',r['tasks'][1]['reasons'])
    def test_monitor_no_override(self):
        e=self.env();e.config.coordinator_mode='monitor';bw=np.arange(6)/10;cpu=bw[::-1]
        b,c,r=e.coordinator.allocate(e.encode()[0][0],bw,cpu)
        np.testing.assert_array_equal(b,bw);np.testing.assert_array_equal(c,cpu)
    def test_invalid_allocation_does_not_advance(self):
        e=self.env()
        with self.assertRaises(ValueError):e.step_candidate(np.zeros(6),[-1]*6,np.ones(6))
        self.assertEqual(e.core.t,0);self.assertEqual(e.pipeline.now,0)
    def test_shared_shield_keeps_ev_and_dr_constraints(self):
        e=self.env();p=ObservedMPC(e.config,e.spec,e.flex,2);rows=[]
        obs,_=e.encode()
        for _ in range(e.config.horizon):
            a,m=p.propose(obs[0]);obs,_,_,_,i=e.step_candidate(a,np.ones(6),np.ones(6));rows.append(i)
            self.assertEqual(i['constraint_violations'],0)
        self.assertAlmostEqual(sum(i['ev_unmet_kwh'] for i in rows),0)
        self.assertAlmostEqual(rows[-1]['dr_backlog_kwh'],0)
    def test_joint_baseline_not_mislabelled(self):
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        import torch
        from vpp_mappo.cyber_compare import compare
        e=self.env();e.config.cyber_mode='joint'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);torch.save({'config':asdict(e.config)},path/'joint.pt')
            with self.assertRaisesRegex(ValueError,'cyber_mode=fixed'):
                compare(path/'joint.pt',path/'result',episodes=1)
            self.assertFalse((path/'result').exists())

if __name__=='__main__':unittest.main()
