"""新增集中控制、因果数据、资源消融与备用激活契约的回归测试。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from vpp_mappo.config import Config
from vpp_research.central_agent import CentralPPO
from vpp_research.environment import ResearchEnv
from vpp_research.ev_import import convert
from vpp_research.timing import TimingContract


class Stage9Tests(unittest.TestCase):
    def test_joint_probability_is_sum(self):
        a=CentralPPO(4,3);o=torch.randn(3,4);w=torch.tensor([1.,0.,0.]);d=a.distribution(o,w);x=d.sample()
        expected=d.base.log_prob(x.squeeze(-1)).sum()
        self.assertTrue(torch.allclose(d.log_prob(x).squeeze(),expected.expand(3)))
        self.assertEqual(tuple(a.value(o,w).shape),(3,))

    def test_resource_ablation_masks_only_selected_controls(self):
        c=Config.load('configs/research_smoke.json');c.cyber_mode='joint';c.coordinator_mode='off'
        for mode in ('control','communication','computation','joint'):
            e=ResearchEnv(c,resource_mode=mode);e.reset(10)
            with patch.object(e,'step_candidate',return_value=None) as call:
                e.step(np.full((18,1),.2));_,bw,cpu=call.call_args.args
                self.assertEqual(bool(np.all(bw==1)),mode in ('control','computation'))
                self.assertEqual(bool(np.all(cpu==1)),mode in ('control','communication'))

    def test_projection_presolve_false_infeasibility_regression(self):
        c=Config.load('configs/research_smoke.json');c.horizon=24;c.dt_hours=1.;c.reserve_hours=1.
        e=ResearchEnv(c);e.reset(10);e.core.sessions=[];e.core.remaining={};e.core.t=22
        e.core.soc=np.array([.9,0.]);e.core.backlog=0.;e.core.shifted=289.57934314617495;e.core.shed_used=100.
        row=dict(load_kw=1129.2951284942267,pv_kw=0.,wind_kw=283.0609576728224,price=.0517)
        for k,v in row.items():e.core.profiles[k][:]=v
        proposal=np.array([-230,150,95,45,0,150.])
        p,meta=e.core.plan(proposal=proposal)
        self.assertLess(meta['solver_residual'],1e-5)
        self.assertEqual(e.core.check(e.core.row(),p[0]['action'],p[0]['ev_kw']),0)
        self.assertEqual(e.core.t,22)

    def test_real_checkpoint_loading_does_not_require_synthetic_carbon(self):
        import csv
        from vpp_research.train import train,load
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'profiles.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['scenario','step','load_kw','pv_kw','wind_kw','price','carbon_g_per_kwh'])
                for t in range(8):w.writerow(['real_day',t,1200,100,100,.1,200])
            c=Config.load('configs/research_smoke.json');c.episodes=1;c.synthetic_carbon_g_per_kwh=None;c.train_csv=str(p)
            c._ev_bundle=dict(schema_version=1,energy_basis='grid_kwh',source='测试曲线',horizon=8,dt_hours=.25,scenarios={'real_day':[]})
            train(c,None,Path(folder)/'model','ordinary')
            state,loaded,agent=load(Path(folder)/'model/latest.pt')
            self.assertIsNone(loaded.synthetic_carbon_g_per_kwh)
            self.assertEqual(state['train_scenarios'],['real_day'])

    def test_ac_reserve_checkpoint_version_and_reload(self):
        from vpp_research.train import train,load
        from vpp_research.environment import AC_OBJECTIVE_VERSION
        with tempfile.TemporaryDirectory() as folder:
            c=Config.load('configs/research_smoke.json');c.episodes=1
            train(c,None,Path(folder)/'model','pareto',reserve_mode='ac_checked')
            state,_,_=load(Path(folder)/'model/latest.pt')
            self.assertEqual(state['objective_version'],AC_OBJECTIVE_VERSION)
            self.assertEqual(state['reserve_mode'],'ac_checked')

    def test_grid_target_is_enforced_without_state_mutation(self):
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c);e.reset(10)
        plans,_=e.core.plan();from vpp_mappo.flex_resources import grid_power
        target=grid_power(e.core.row(),plans[0]['action']);other,_=e.core.plan(grid_target=target)
        self.assertAlmostEqual(grid_power(e.core.row(),other[0]['action']),target,places=5)
        self.assertEqual(e.core.t,0)

    def test_online_carbon_does_not_read_current_actual(self):
        e=ResearchEnv(Config.load('configs/research_smoke.json'));e.reset(2)
        e.core.profiles['carbon_g_per_kwh']=np.full(8,999.)
        e.core.profiles['carbon_observed_g_per_kwh']=np.full(8,111.)
        self.assertEqual(e.sensor_payloads()[2]['carbon_g_per_kwh'],111.)

    def test_future_ev_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.json';p.write_text(json.dumps({'_items':[dict(sessionID='a',connectionTime='2019-01-01T08:00:00Z',userInputs=[dict(modifiedAt='2019-01-01T09:00:00Z',requestedDeparture='2019-01-01T10:00:00Z',kWhRequested=4)])]}))
            with self.assertRaises(ValueError):convert(p,Path(d)/'out.json')

    def test_delayed_command_and_compute_budget(self):
        from vpp_research.timing import CommandQueue
        contract=TimingContract(downlink_bps=10000,deadline_seconds=1800)
        q=CommandQueue(contract);q.submit(0,np.arange(6),.1,.1)
        self.assertIsNone(q.receive(0)[0]);self.assertTrue(np.array_equal(q.receive(900)[0],np.arange(6)))
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c,timing_contract=contract,control_cycles=1.e9)
        e.reset(3);*_,info=e.step(np.zeros((6,1)))
        self.assertTrue(info['local_fallback']);self.assertLess(info['dt_available_cpu_cycles_per_second'],e.cyber_spec.cpu_cycles_per_second)
        self.assertEqual(info['control_cycles'],1.e9)

    def test_recalibration_cannot_consume_future_labels(self):
        from vpp_research.recalibration import DelayedBlockCalibration
        c=DelayedBlockCalibration(np.ones(9),[1.]*12)
        c.submit(100.,2)
        self.assertTrue(np.all(c.widths(1)==1.))
        c.submit(100.,2)
        self.assertTrue(np.all(c.widths(2)==100.))
        with self.assertRaises(ValueError):c.widths(1)

    def test_deadline_and_outage_accounting(self):
        self.assertTrue(TimingContract(downlink_bps=0).latency(.1,.1)['deadline_missed'])
        self.assertTrue(TimingContract(deadline_seconds=.1).latency(.1,.1)['deadline_missed'])
        with self.assertRaises(ValueError):TimingContract().latency(-1,0)

if __name__=='__main__':unittest.main()
