"""v4.1 的求解复核、证书保持、故障后备和公开时序回归。"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.optimization import DispatchInfeasible
from vpp_research.environment import ResearchEnv
from vpp_research.safety import guard
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_research.delayed_mpc import arrival_observation


def environment():
    c=Config.load('configs/research_smoke.json');c.safety_revision=1;c.research_version=4;c.horizon=4
    e=ResearchEnv(c,reserve_mode='pcc_checked');e.reset(72)
    return e


class V41Tests(unittest.TestCase):
    def test_extreme_objective_presolve_retry(self):
        import vpp_mappo.flex_optimization as module
        e=environment();real=module.milp;calls=[]
        def solve(*args,**kwargs):
            calls.append(kwargs['options'])
            if len(calls)==1:return SimpleNamespace(status=2,x=None,message='模拟预处理误报')
            return real(*args,**kwargs)
        with patch.object(module,'milp',solve):plans,meta=e.core.plan(objective='max_grid')
        self.assertTrue(meta['presolve_retry']);self.assertFalse(calls[1]['presolve'])
        self.assertEqual(e.core.check(e.core.row(),plans[0]['action'],plans[0]['ev_kw']),0)

    def test_feasible_and_infeasible_interval(self):
        e=environment();x=e.encode()[0][0];x[9]=.2;x[11:13]=0;x[54:63]=0
        action,meta,certificate=guard(np.zeros(6),x,e.spec,e.flex,e.network,e.config.dt_hours,e.config.horizon,diagnostics=True)
        self.assertTrue(meta['guard_feasible']);self.assertTrue(certificate(action))
        x[59]=10
        _,bad,cert=guard(np.zeros(6),x,e.spec,e.flex,e.network,e.config.dt_hours,e.config.horizon,diagnostics=True)
        self.assertFalse(bad['guard_feasible']);self.assertFalse(cert(action))
        self.assertGreater(bad['guard_diagnostic_relaxation'],0);self.assertFalse(bad['guard_relaxation_used_for_execution'])

    def test_interval_constraints_survive_local_projection(self):
        e=environment();matrix=np.array([[1.,0,0,0,0,0],[-1.,0,0,0,0,0]])
        bounds=[(-500,500),(0,300),(-150,100),(0,50),(0,5000),(0,5000)]
        e.core.guard_constraints=(matrix,np.zeros(2),bounds)
        *_,i=e.core.step_physical(np.array([200.,0,0,0,0,0]))
        self.assertAlmostEqual(i['ess_power_kw'],0,places=5);self.assertEqual(i['safety_mode'],'interval_projected');self.assertTrue(i['safety_hard_verified'])

    def test_timeout_and_service_degradation(self):
        e=environment()
        for s in e.core.active():e.core.remaining[s['id']]=100000.
        with patch.object(e.core,'plan',side_effect=DispatchInfeasible('故障注入：超时',reason='timeout')):
            *_,i=e.core.step_physical(np.zeros(6))
        self.assertTrue(i['emergency']);self.assertTrue(i['emergency_service_degraded']);self.assertTrue(i['safety_hard_verified'])
        self.assertTrue(any(r['reason']=='timeout' for r in i['safety_failures']))

    def test_ac_failure_refuses_state_advance(self):
        e=environment();soc=e.core.soc.copy()
        with patch.object(e.core.network,'audit',return_value=dict(ac_converged=False,ac_violations=1)),self.assertRaises(DispatchInfeasible):e.core.step_physical(np.zeros(6))
        self.assertEqual(e.core.t,0);np.testing.assert_array_equal(e.core.soc,soc);self.assertTrue(e.core.ac_rejected)

    def test_delay_prediction_zero_delay_and_expiration(self):
        e=environment();c=e.config;c.command_timing=True;c.control_cycles=0;c.downlink_propagation_seconds=.02
        p=ObservedMPC(c,e.spec,e.flex,2);x=e.encode()[0][0].copy();x[0]=1/c.horizon;x[78]=.1;x[77]=0;x[71]=.05
        y,m=arrival_observation(p,x);self.assertEqual(m['mpc_delay_steps'],1);self.assertLess(y[1],x[1]);self.assertEqual(m['mpc_consumed_command_boundaries'],[1])
        c.command_timing=False;y,m=arrival_observation(p,x);self.assertEqual(m['mpc_delay_steps'],0);self.assertEqual(y[1],x[1])
        c.command_timing=True;c.command_deadline_steps=.5;y,m=arrival_observation(p,x);self.assertIsNone(y);self.assertEqual(m['mpc_unavailable_reason'],'expired_before_dispatch_boundary')

    def test_queued_commands_consumed_once(self):
        e=environment();c=e.config;c.horizon=8;c.command_timing=True;c.control_cycles=0;c.downlink_propagation_seconds=c.dt_hours*3600*1.2;c.command_deadline_steps=3
        p=ObservedMPC(c,e.spec,e.flex,2);x=e.encode()[0][0].copy();x[0]=2/c.horizon;x[78]=.2
        p.issued_commands=[(0,2,np.array([20.,0,0,0,0,0])),(1,3,np.array([-20.,0,0,0,0,0]))]
        y,m=arrival_observation(p,x);self.assertEqual(m['mpc_consumed_command_boundaries'],[2,3]);self.assertFalse(m['mpc_queue_information_incomplete'])

if __name__=='__main__':unittest.main()
