"""残差拟合隔离、向量更新、固定协议与区间保护的独立验证。"""
import unittest
import numpy as np
import torch
from vpp_mappo.config import Config
from vpp_research.dt import fit,predict,assess,TARGETS
from vpp_research.pareto_agent import ParetoAgent,vector_gae
from vpp_research.environment import ResearchEnv
from vpp_research.safety import guard


class Stage8Tests(unittest.TestCase):
    def rows(self,start,count):
        rows=[]
        for i in range(start,start+count):
            x=np.zeros(54);x[1]=.5;x[9]=.3+i*.001
            y=x[TARGETS].copy();y[5]+=.02
            rows.append(dict(scenario=str(i),x=x.tolist(),y=y.tolist()))
        return rows
    def test_calibration_and_fit_separation(self):
        with self.assertRaises(ValueError):fit(self.rows(0,12),self.rows(0,12))
        with self.assertRaises(ValueError):fit(self.rows(0,12),self.rows(20,3),alpha=.1)
        m=fit(self.rows(0,20),self.rows(30,12))
        self.assertEqual(m['calibration_rank'],12)
        with self.assertRaises(ValueError):assess(m,self.rows(0,2))
    def test_residual_recovers_systematic_bias(self):
        m=fit(self.rows(0,20),self.rows(30,12));r=assess(m,self.rows(50,5))
        self.assertLess(r['rmse'],1e-5)
    def test_vector_gae_analytic_terminal(self):
        r=np.array([[1,2,3],[4,5,6]])
        a,ret=vector_gae(r,np.zeros((3,3)),[0,1],gamma=1,lam=1)
        np.testing.assert_allclose(a,[[5,7,9],[4,5,6]])
        a,_=vector_gae(r,np.ones((3,3))*100,[1,1],gamma=1,lam=1)
        np.testing.assert_allclose(a,r-100)
    def test_preference_actor_and_vector_critic(self):
        torch.manual_seed(1);p=ParetoAgent(65,6);obs=np.zeros((6,65),np.float32)
        a,lp,v=p.act(obs,[1,0,0],True);b,_,u=p.act(obs,[0,1,0],True)
        self.assertEqual(v.shape,(3,));self.assertFalse(np.array_equal(a,b))
        self.assertEqual(lp.shape,(6,))
    def test_vector_update_changes_parameters(self):
        torch.manual_seed(2);p=ParetoAgent(65,6);roll={k:[] for k in ('obs','preferences','actions','logprobs','values','vectors','dones')}
        for t in range(4):
            obs=np.ones((6,65),np.float32)*t/10;w=np.array([.2,.3,.5]);a,lp,v=p.act(obs,w)
            for k,item in zip(roll,(obs,w,a,lp,v,np.array([1.,t,2.]),t==3)):roll[k].append(item)
        before=torch.cat([q.detach().flatten().clone() for q in p.parameters()]);p.update(roll)
        after=torch.cat([q.detach().flatten() for q in p.parameters()]);self.assertGreater(float(torch.linalg.norm(before-after)),0)
    def test_observation_never_reads_hidden_truth(self):
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c);e.reset(40);a=e.encode()[0]
        e.core.soc[0]=.8;e.core.profiles['wind_kw'][0]+=90
        np.testing.assert_array_equal(a,e.encode()[0]);self.assertEqual(a.shape,(6,65))
    def test_objective_cost_carbon_units(self):
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c);e.reset(50)
        *_,i=e.step(np.zeros((6,1)))
        self.assertAlmostEqual(i['objective_vector'][0],-(i['cost']+i['terminal_penalty'])/100)
        expected=max(i['grid_power_kw'],0)*c.dt_hours*.3+i['cyber_energy_j']/3.6e6*.3
        self.assertAlmostEqual(i['carbon_kg'],expected)
    def test_zero_width_guard_and_execution_certificate(self):
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c);x=e.reset(1)[0][0]
        a,m,cert=guard(np.zeros(6),x,e.spec,e.flex,e.network,c.dt_hours,c.horizon)
        self.assertTrue(m['guard_feasible']);self.assertTrue(cert(a))
        self.assertFalse(cert(np.array([0,0,0,0,100000,0])))
    def test_ac_rejection_does_not_execute_physical_step(self):
        from vpp_mappo.optimization import DispatchInfeasible
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c,ac_safe=True,vector_metrics=False);e.reset(1)
        e.core.network.audit=lambda row,action:dict(ac_converged=True,ac_violations=1)
        with self.assertRaises(DispatchInfeasible):e.step(np.zeros((6,1)))
        self.assertEqual(e.core.t,0)
        with self.assertRaises(DispatchInfeasible):e.core.step_physical(np.zeros(6))

    def test_inconsistent_interval_reports_failure(self):
        c=Config.load('configs/research_smoke.json');e=ResearchEnv(c);x=e.reset(1)[0][0];x[1]=3.;x[54]=.1
        a,m,cert=guard(np.zeros(6),x,e.spec,e.flex,e.network,c.dt_hours,c.horizon)
        self.assertFalse(m['guard_feasible']);self.assertFalse(cert(a))

if __name__=='__main__':unittest.main()
