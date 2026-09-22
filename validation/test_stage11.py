"""PCC 口径、紧急后备和旧版本隔离的行为验证。"""
import unittest
from unittest.mock import patch
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.optimization import DispatchInfeasible
from vpp_research.ac_safety import ACCheckedFlex
from vpp_research.environment import ResearchEnv, PCC_OBJECTIVE_VERSION
from vpp_research.pcc_reserve import target_plan, activate, checked_capacity


class PCCSafetyTests(unittest.TestCase):
    def setUp(self):
        self.config=Config.load('configs/research_smoke.json')
        self.core=ACCheckedFlex(self.config, emergency=True);self.core.reset(19)

    def test_actual_pcc_tracking_includes_losses_and_does_not_change_state(self):
        e=self.core;p,_=e.plan();audit=e.network.audit(e.row(),p[0]['action']);target=audit['ac_grid_kw']+10
        soc=e.soc.copy();p,meta=target_plan(e,target)
        self.assertLessEqual(abs(meta['pcc_error_kw']),.1)
        self.assertGreater(meta['ac_loss_kw'],1)
        self.assertEqual(e.t,0);np.testing.assert_array_equal(e.soc,soc)
        # 对已验证计划不再投影改动 EV 分配或功率目标。
        with patch.object(e,'plan',side_effect=AssertionError('不可再次投影')):
            *_,info=e.step_physical(p)
        self.assertLessEqual(abs(info['ac_grid_kw']-target),.1)
        self.assertGreater(abs(info['grid_power_kw']-target),1)

    def test_multistep_activation_is_verified_on_actual_trajectory(self):
        e=self.core;p,_=e.plan();g=e.network.audit(e.row(),p[0]['action'])['ac_grid_kw']
        r=activate(e,g,10.,.5,duration_steps=2)
        self.assertFalse(r['failed'],r)
        self.assertEqual(len(r['steps']),2)
        self.assertTrue(all(s['error_kw']<=.1 for s in r['steps']))
        self.assertEqual(e.t,0)
        with self.assertRaises(ValueError):activate(e,g,10,1,duration_steps=100)

    def test_solver_failure_uses_audited_emergency_action(self):
        e=self.core
        with patch.object(e,'plan',side_effect=DispatchInfeasible('模拟求解超时')):
            *_,info=e.step_physical(np.zeros(6))
        self.assertTrue(info['emergency']);self.assertEqual(info['ac_violations'],0)
        self.assertEqual(info['emergency_hard_violations'],0);self.assertEqual(e.t,1)
        self.assertTrue(e.config.safety)

    def test_service_shortfall_is_reported_not_hidden(self):
        e=self.core;e.t=self.config.horizon-1
        e.remaining['demo_0']=100.
        with patch.object(e,'plan',side_effect=DispatchInfeasible('需求超出能力')):
            *_,info=e.step_physical(np.zeros(6))
        self.assertTrue(info['emergency_service_degraded'])
        self.assertGreater(info['ev_unmet_kwh'],0)
        self.assertGreater(info['constraint_violations'],0)
        self.assertEqual(info['emergency_hard_violations'],0)

    def test_no_hard_safe_candidate_does_not_advance(self):
        e=self.core;soc=e.soc.copy()
        with patch.object(e,'plan',side_effect=DispatchInfeasible('超时')), patch.object(e.network,'audit',return_value={'ac_converged':False,'ac_violations':1}):
            with self.assertRaises(DispatchInfeasible):e.step_physical(np.zeros(6))
        self.assertEqual(e.t,0);np.testing.assert_array_equal(e.soc,soc)
        self.assertTrue(e.ac_rejected)

    def test_v3_cost_carbon_and_reserve_have_explicit_pcc_version(self):
        e=ResearchEnv(self.config,reserve_mode='pcc_checked');e.reset(19)
        _,_,reward,_,info=e.step(np.zeros((e.num_agents,1)))
        self.assertEqual(info['objective_version'],PCC_OBJECTIVE_VERSION)
        self.assertEqual(info['objective_grid_kw'],info['ac_grid_kw'])
        self.assertEqual(info['objective_cost'],info['ac_cost'])
        self.assertAlmostEqual(info['objective_vector'][0],-(info['ac_cost']+info['terminal_penalty'])/100)
        self.assertAlmostEqual(info['carbon_kg'],info['ac_carbon_kg'])
        self.assertAlmostEqual(float(reward[0,0]),info['objective_vector'][0],places=5)
        confirmation=info['reserve_confirmation']
        self.assertTrue(all(abs(r['error_kw'])<=.1 for r in confirmation['samples']))

    def test_tracking_nonconvergence_fails_explicitly(self):
        e=self.core
        with patch.object(e.network,'audit',return_value={'ac_converged':False,'ac_violations':1}):
            with self.assertRaises(DispatchInfeasible):target_plan(e,1000.)
        self.assertEqual(e.t,0)
        with self.assertRaises(ValueError):checked_capacity(e,1000.,-1.)




class OLSTests(unittest.TestCase):
    def test_exact_small_oracle_discovers_interior_solution(self):
        from vpp_research.ols import next_weight, corner_weights, select_member
        oracle=np.array([[10,0,0],[0,10,0],[0,0,10],[4,4,4]],dtype=float)
        values=[];weights=[]
        for i in range(4):
            w,priority=next_weight(values,weights)
            value=oracle[np.argmax(oracle@w)]
            values.append(value);weights.append(w)
        np.testing.assert_allclose(weights[-1],np.ones(3)/3)
        np.testing.assert_allclose(values[-1],[4,4,4])
        self.assertGreater(priority,0)
        result={'members':[{'index':i,'validation_value':v.tolist()} for i,v in enumerate(values)]}
        self.assertEqual(select_member(result,[1/3]*3)['index'],3)
        with self.assertRaises(ValueError):select_member(result,[1,1,1])

    def test_ols_counts_all_subpolicies_and_separate_selection_budget(self):
        import tempfile
        from pathlib import Path
        from vpp_research.ols import run
        c=Config.load('configs/research_smoke.json')
        def fake_evaluate(checkpoint, preferences, seeds, **kwargs):
            w=np.array(preferences[0]);v=(10*w).tolist()
            return [dict(seed=s,preference=w.tolist(),vector=v,failed=False,env_steps=c.horizon,
                violations=0,ac_violations=0,ac_failed=0,reserve_invalid=0,ev_unmet_kwh=0,dr_backlog_kwh=0) for s in seeds]
        with tempfile.TemporaryDirectory() as d, patch('vpp_research.ols.train') as train_mock, patch('vpp_research.ols.evaluate',side_effect=fake_evaluate):
            result=run(c,None,Path(d)/'ols',total_episodes=12,policies=4)
            self.assertEqual(train_mock.call_count,4)
            self.assertEqual(result['completed_training_steps'],12*c.horizon)
            self.assertEqual(result['completed_selection_steps'],8*c.horizon)
            self.assertEqual(result['status'],'completed')




class DepartureTests(unittest.TestCase):
    def test_early_departure_is_hidden_until_event_and_shortfall_counted_once(self):
        from vpp_mappo.flex_environment import FlexVPPAdapter
        c=Config.load('configs/research_smoke.json')
        c._ev_bundle=dict(schema_version=1,source='合成事件机制测试',energy_basis='grid_kwh',horizon=c.horizon,dt_hours=c.dt_hours,
            scenarios={'synthetic:5':[dict(id='ev',arrival_step=0,departure_step=8,energy_kwh=12.,max_kw=11.)]},
            actual_departure_events={'synthetic:5':[dict(id='ev',departure_step=1)]})
        e=FlexVPPAdapter(c);e.reset(5)
        self.assertEqual(e.sessions[0]['departure_step'],8)
        p,_=e.plan();self.assertEqual(e.sessions[0]['departure_step'],8)
        e.step_physical(p[0])
        self.assertFalse(e.active());self.assertEqual(e.sessions[0]['departure_step'],1)
        p,_=e.plan();*_,info=e.step_physical(p[0])
        self.assertEqual(info['early_ev_departures'],1)
        self.assertGreater(info['early_ev_unmet_kwh'],0)
        self.assertAlmostEqual(info['ev_unmet_kwh'],info['early_ev_unmet_kwh'])
        p,_=e.plan();*_,info=e.step_physical(p[0])
        self.assertEqual(info['early_ev_departures'],0);self.assertEqual(info['ev_unmet_kwh'],0)




class DepartureImportTests(unittest.TestCase):
    def test_actual_event_does_not_replace_declared_deadline(self):
        import json
        import tempfile
        from pathlib import Path
        from vpp_research.ev_import import convert
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.json';out=Path(d)/'out.json'
            raw.write_text(json.dumps({'_items':[dict(sessionID='x',connectionTime='2019-01-01T08:00:00Z',
                disconnectTime='2019-01-01T09:00:00Z',userInputs=[dict(modifiedAt='2019-01-01T08:00:00Z',
                requestedDeparture='2019-01-01T10:00:00Z',kWhRequested=4)])]}))
            convert(raw,out,replay_actual_departures=True)
            bundle=json.loads(out.read_text())
            self.assertEqual(bundle['scenarios']['2019-01-01'][0]['departure_step'],40)
            self.assertEqual(bundle['actual_departure_events']['2019-01-01'][0]['departure_step'],36)




class DRDataTests(unittest.TestCase):
    def test_negative_response_does_not_become_guaranteed_positive_capacity(self):
        from dataclasses import asdict
        from vpp_mappo.flex_resources import FlexSpec
        from vpp_research.dr_import import conservative_flex
        base=asdict(FlexSpec())
        record,meta=conservative_flex(base,{'summary':{'train_ratio_q10':-.1},'source':'测试'})
        self.assertEqual(record['dr_shed_kw'],0)
        self.assertEqual(record['dr_shed_budget_kwh'],0)
        self.assertEqual(record['dr_repay_kw'],base['dr_repay_kw'])
        self.assertFalse(meta['certified']);self.assertGreater(base['dr_shed_kw'],0)


if __name__=='__main__':unittest.main()
