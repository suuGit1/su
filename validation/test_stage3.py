"""统一优化、网络物理与真实数据转换的回归验证。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from vpp_mappo.config import Config
from vpp_mappo.dispatch import DispatchSpec
from vpp_mappo.network import Network33
from vpp_mappo.optimization import solve_dispatch, DispatchInfeasible
from vpp_mappo.grid_environment import GridVPPAdapter
from vpp_mappo.prepare_opsd import prepare, COLUMNS
from vpp_mappo.data import CSVProfiles


class Stage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=DispatchSpec.load(); cls.net=Network33(cls.spec)

    def test_case_and_ac_balance(self):
        self.assertEqual(len(self.net.net.bus),33); self.assertEqual(len(self.net.lines),32)
        self.assertAlmostEqual(self.net.base_p.sum(),3.715,places=6)
        row=dict(load_kw=1800.,pv_kw=300.,wind_kw=100.,price=.1); action=np.array([100.,-50.,80.])
        audit=self.net.audit(row,action)
        self.assertTrue(audit['ac_converged'])
        self.assertAlmostEqual(audit['ac_grid_kw'],1800-300-100-sum(action)+audit['ac_loss_kw'],delta=1e-3)

    def test_linear_affine_consistency(self):
        row=dict(load_kw=1800.,pv_kw=300.,wind_kw=100.,price=.1); a=np.array([100.,-50.,80.])
        p,q=self.net.injections(row,a); bp,bq=self.net.downstream@p,self.net.downstream@q
        v2=1-2*self.net.paths@(self.net.r*bp+self.net.x*bq)/self.net.kv**2
        mat,lo,hi=self.net.affine(row)
        expected=self.spec.voltage_min**2-(lo[:33]-mat[:33]@a)
        np.testing.assert_allclose(v2,expected,atol=1e-12)

    def test_optimizer_objective_equals_executor(self):
        cfg=Config(network_model='ieee33',horizon=4,dt_hours=1.)
        env=GridVPPAdapter(cfg); env.reset(2)
        rows=[env.row(t) for t in range(4)]
        actions,meta=solve_dispatch(env.spec,env.network,rows,env.soc,1.,cfg.terminal_soc_penalty)
        total=0
        for a in actions:
            info=env.step_physical(a)[-1]
            self.assertEqual(info['constraint_violations'],0)
            self.assertLess(info['shield_l1_kw'],1e-5)
            total+=info['cost']+info['terminal_penalty']
        self.assertTrue(meta['solver_optimal']); self.assertAlmostEqual(meta['solver_objective'],total,places=5)

    def test_negative_price_has_no_hidden_arbitrage(self):
        rows=[dict(load_kw=600.,pv_kw=300.,wind_kw=100.,price=-.3) for _ in range(2)]
        actions,meta=solve_dispatch(self.spec,self.net,rows,self.spec.initial_soc,1.,1000.)
        soc=np.array(self.spec.initial_soc); total=0
        for row,a in zip(rows,actions):
            grid=row['load_kw']-row['pv_kw']-row['wind_kw']-sum(a)
            total+=self.spec.cost(grid,a,row['price'],1)
            soc=self.spec.next_soc(soc,a,1)
        total+=1000*np.maximum(np.array(self.spec.target_soc)-soc,0).sum()
        self.assertAlmostEqual(total,meta['solver_objective'],places=5)

    def test_infeasible_is_explicit(self):
        row=dict(load_kw=100000.,pv_kw=0.,wind_kw=0.,price=.1)
        with self.assertRaises(DispatchInfeasible):
            solve_dispatch(self.spec,self.net,[row],self.spec.initial_soc,1.,1000.)

    def test_shield_shared_soc_constraints(self):
        cfg=Config(network_model='ieee33',horizon=2,dt_hours=1.)
        env=GridVPPAdapter(cfg); env.reset(1); env.soc=np.array(env.spec.soc_min)
        info=env.step_physical(np.array([250.,300.,0.]))[-1]
        self.assertGreater(info['pre_shield_violations'],0)
        self.assertEqual(info['constraint_violations'],0)
        self.assertGreater(info['shield_l1_kw'],0)

    def test_opsd_split_scaling_and_partial_overlap_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw.csv'; out=Path(tmp)/'out'
            times=pd.date_range('2019-01-01',periods=72,freq='h',tz='UTC')
            frame=pd.DataFrame({'utc_timestamp':times})
            for k,col in COLUMNS.items(): frame[col]=30. if k=='price' else np.repeat([10.,20.,100.],24)
            frame.to_csv(raw,index=False)
            meta=prepare(raw,out,'2019-01-01','2019-01-02','2019-01-03','2019-01-04')
            train=CSVProfiles(out/'train.csv',24); test=CSVProfiles(out/'test.csv',24)
            self.assertEqual(train.profiles[0]['load_kw'][0],1800)
            self.assertEqual(test.profiles[0]['load_kw'][0],18000)
            self.assertAlmostEqual(test.profiles[0]['price'][0],.03)
            self.assertFalse(set(train.scenario_names)&set(test.scenario_names))
            frame=pd.read_csv(out/'train.csv'); frame.scenario='renamed'; frame.to_csv(Path(tmp)/'renamed.csv',index=False)
            self.assertEqual(train.fingerprints,CSVProfiles(Path(tmp)/'renamed.csv',24).fingerprints)

    def test_unsupported_information_asymmetry_rejected(self):
        with self.assertRaises(ValueError): GridVPPAdapter(Config(network_model='ieee33',delay_steps=1))

    def test_resource_snapshot_survives_source_removal(self):
        from vpp_mappo.runner import train, evaluate
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'spec.json'
            record=self.spec.record(); record['power_max']=[200.,250.]
            path.write_text(json.dumps(record))
            cfg=Config(network_model='ieee33',dispatch_spec=str(path),episodes=1,horizon=2,hidden_size=16,ppo_epoch=1)
            train(cfg,Path(tmp)/'train')
            path.unlink()
            rows=evaluate(Path(tmp)/'train/latest.pt',Path(tmp)/'test',episodes=1)
            self.assertEqual(rows[0]['violations'],0)


if __name__=='__main__': unittest.main(verbosity=2)
