"""会话模型守恒、信息边界与统一优化回归。"""
import unittest
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.environment import VPPAdapter
from vpp_mappo.flex_resources import validate_sessions, grid_power
from vpp_mappo.optimization import DispatchInfeasible

class SessionsTests(unittest.TestCase):
    def env(self):
        c=Config.load('configs/sessions_smoke.json');e=VPPAdapter(c);e.reset(1);return e
    def test_oracle_replay(self):
        e=self.env();p,m=e.plan(oracle=True);rows=[]
        for a in p:
            *_,i=e.step_physical(a);rows.append(i)
            self.assertEqual(i['constraint_violations'],0)
            self.assertEqual(i['shield_solver_seconds'],0)
        self.assertAlmostEqual(sum(r['cost']+r['terminal_penalty'] for r in rows),m['solver_objective'],places=5)
        self.assertAlmostEqual(sum(r['ev_unmet_kwh'] for r in rows),0)
        self.assertAlmostEqual(sum(r['ev_delivered_kwh'] for r in rows),sum(s['energy_kwh'] for s in e.sessions))
        self.assertAlmostEqual(e.backlog,0)
        self.assertAlmostEqual(sum(r['dr_shifted_kwh']-r['dr_repaid_kwh'] for r in rows),0)
    def test_causal_forecast(self):
        a=self.env();b=self.env();b.profiles={k:v.copy() for k,v in b.profiles.items()}
        for k in b.profiles:b.profiles[k][1:]*=1.3
        b.sessions[1]=dict(b.sessions[1],energy_kwh=1);b.remaining[b.sessions[1]['id']]=1
        np.testing.assert_allclose(a.encode()[0],b.encode()[0])
        np.testing.assert_allclose(a.plan(lookahead=2)[0][0]['action'],b.plan(lookahead=2)[0][0]['action'])
    def test_mpc_deadlines(self):
        e=self.env()
        for _ in range(e.config.horizon):
            p,_=e.plan(lookahead=2);*_,i=e.step_physical(p[0]);self.assertEqual(i['constraint_violations'],0);self.assertAlmostEqual(i['ev_unmet_kwh'],0)
        self.assertAlmostEqual(e.backlog,0)
    def test_raw_safety(self):
        e=self.env()
        for _ in range(e.config.horizon):
            *_,i=e.step(np.ones((6,1))*3);self.assertEqual(i['constraint_violations'],0);self.assertAlmostEqual(i['ev_unmet_kwh'],0)
        self.assertLessEqual(e.shed_used,e.flex.dr_shed_budget_kwh+1e-5)
        self.assertLessEqual(e.shifted,e.flex.dr_shift_budget_kwh+1e-5)
    def test_curtailment_balance(self):
        e=self.env();r=e.row();a=np.array([0,0,0,0,r['pv_kw'],r['wind_kw']]);self.assertAlmostEqual(grid_power(r,a),r['load_kw'])
        self.assertEqual(e.network.pa.shape,(33,6))
    def test_inactive_charge_rejected(self):
        e=self.env();alloc={s['id']:0 for s in e.sessions};alloc[e.sessions[1]['id']]=1
        self.assertGreater(e.check(e.row(),np.array([0,1,0,0,0,0]),alloc),0)
    def test_impossible_station(self):
        e=self.env();e.flex.ev_station_kw=.001
        with self.assertRaises(DispatchInfeasible):e.plan(oracle=True)
    def test_invalid_session(self):
        with self.assertRaises(ValueError):validate_sessions([dict(id='a',arrival_step=0,departure_step=1,energy_kwh=10,max_kw=1)],8,.25)

if __name__=='__main__':unittest.main()
