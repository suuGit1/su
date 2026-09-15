"""三目标口径、普通 MAPPO 保留、非支配档案和并行接口验证。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import json
import tempfile
import unittest
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.grid_environment import GridVPPAdapter
from vpp_mappo.algorithms import REGISTRY
from vpp_mappo.objectives import carbon_kg, OBJECTIVE_NAMES, reserve_envelope
from vpp_mappo.pareto import non_dominated, hypervolume, igd
from vpp_mappo.data import CSVProfiles


class Stage4Tests(unittest.TestCase):
    def cfg(self, **kwargs):
        base=Config(network_model='ieee33',metrics_enabled=True,synthetic_carbon_g_per_kwh=300,
                    horizon=2,episodes=1,hidden_size=16,ppo_epoch=1,gamma=1.0)
        return replace(base,**kwargs).validate()

    def test_ordinary_mappo_preserved(self):
        self.assertTrue(REGISTRY['mappo']); self.assertTrue(REGISTRY['weighted_mappo'])
        self.assertNotIn('pareto_mappo',REGISTRY)
        with self.assertRaises(ValueError): self.cfg(algorithm='pareto_mappo')
        base=self.cfg(algorithm='mappo',objective_weights=(0.,0.,1.))
        a=GridVPPAdapter(base); b=GridVPPAdapter(replace(base,metrics_enabled=False))
        a.reset(3);b.reset(3)
        for _ in range(2):
            action=np.array([[.1],[-.2],[.3]])
            ia=a.step(action)[-1];ib=b.step(action)[-1]
            self.assertAlmostEqual(ia['reward'],ib['reward'],places=10)
            self.assertAlmostEqual(ia['cost'],ib['cost'],places=10)

    def test_economic_weight_matches_ordinary_reward(self):
        a=GridVPPAdapter(self.cfg());b=GridVPPAdapter(self.cfg(algorithm='weighted_mappo'))
        a.reset(4);b.reset(4)
        for _ in range(2):
            ia=a.step(np.zeros((3,1)))[-1];ib=b.step(np.zeros((3,1)))[-1]
            self.assertAlmostEqual(ia['reward'],ib['reward'],places=10)
        self.assertAlmostEqual(ia['economic_return'],-(ia['cost']+ia['terminal_penalty'])/100)

    def test_carbon_boundary_and_units(self):
        self.assertEqual(carbon_kg(1000,.25,400),100.)
        self.assertEqual(carbon_kg(-1000,.25,400),0.)
        with self.assertRaises(ValueError): carbon_kg(1,1,-1)

    def test_no_implicit_carbon_for_real_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.csv';p.write_text('scenario,step,load_kw,pv_kw,wind_kw,price\nd,0,1000,0,0,.1\nd,1,1000,0,0,.1\n')
            with self.assertRaises(ValueError):GridVPPAdapter(self.cfg(),p)
        with self.assertRaises(ValueError):GridVPPAdapter(self.cfg(synthetic_carbon_g_per_kwh=None))

    def test_reserve_respects_soc(self):
        env=GridVPPAdapter(self.cfg());env.reset(3);row=env.row()
        low=reserve_envelope(env.spec,env.network,row,env.spec.soc_min,1500,1.,30.)
        high=reserve_envelope(env.spec,env.network,row,env.spec.soc_max,1500,1.,30.)
        self.assertGreater(low['reserve_min_grid_kw'],high['reserve_min_grid_kw'])
        self.assertGreater(low['reserve_max_grid_kw'],high['reserve_max_grid_kw'])

    def test_pareto_ties_feasibility_and_hv(self):
        points=[[2,1,1],[1,2,1],[1,1,1],[2,1,1],[9,9,9]]
        indices=non_dominated(points,[True,True,True,True,False])
        self.assertEqual(indices,[0,1])
        self.assertAlmostEqual(hypervolume(np.array(points)[indices],[0,0,0]),3.)
        self.assertEqual(hypervolume([],[-1,-1,0]),0.)
        self.assertEqual(igd([[1,1,1]],[[1,1,1]]),0.)
        with self.assertRaises(ValueError):igd([[1,1,1]],[])

    def test_parallel_vector_sum_and_termination(self):
        from vpp_mappo.mo_parallel import VPPMOParallelEnv
        env=VPPMOParallelEnv(self.cfg());obs,_=env.reset(seed=9)
        self.assertEqual(env.reward_space('ess').shape,(3,))
        self.assertEqual(env.state().shape,(39,))
        for _ in range(2):
            obs,r,terminated,truncated,info=env.step({a:np.zeros(1) for a in env.agents})
            np.testing.assert_allclose(sum(r.values()),info['ess']['team_vector'],rtol=1e-6)
        self.assertTrue(all(terminated.values()));self.assertFalse(any(truncated.values()))
        self.assertEqual(env.agents,[]);self.assertEqual(env.step({}),({},{},{},{},{}))


if __name__=='__main__':unittest.main(verbosity=2)
