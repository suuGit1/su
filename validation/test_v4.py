"""验证跨回合GAE、PopArt单位、公开观察、四类学习器及中断预算。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from vpp_mappo.config import Config
from vpp_research.pareto_agent import vector_gae
from vpp_research.pareto_v4 import ParetoV4
from vpp_research.environment import ResearchEnv
from vpp_research.control_v4 import ess_bounds,decode

class V4Tests(unittest.TestCase):
    def test_terminal_blocks_next_day_gae(self):
        r=np.array([[1,2,3],[100,200,300]],dtype=float);a,_=vector_gae(r,np.zeros((3,3)),[True,True]);np.testing.assert_allclose(a,r)

    def test_popart_preserves_unnormalized_prediction(self):
        torch.manual_seed(7);a=ParetoV4(80,6,16);x=torch.randn(8,6,80);w=torch.ones(8,3)/3
        before=a.value(x,w).detach().clone();a.update_statistics(torch.randn(8,3)*20+100)
        torch.testing.assert_close(before,a.value(x,w),atol=2e-5,rtol=2e-5)
        self.assertFalse(set(map(id,a.actor.parameters())) & set(map(id,a.critic.parameters())))

    def test_queue_observation_no_physical_soc_leak(self):
        c=Config.load('configs/research_smoke.json');c.research_version=4;c.command_timing=True
        env=ResearchEnv(c);o,s=env.reset(99);self.assertEqual(o.shape,(env.num_agents,80));self.assertEqual(s.shape,(env.num_agents,80*env.num_agents))
        env.command_queue.submit(0,np.array([80,0,0,0,0,0]),.1,0);queued=env.encode()[0];self.assertNotEqual(queued[0,71],o[0,71])
        env.core.soc[0]=.12345;np.testing.assert_array_equal(queued,env.encode()[0])

    def test_estimated_bounds_and_low_curtailment(self):
        c=Config.load('configs/research_smoke.json');c.research_version=4;env=ResearchEnv(c);obs,_=env.reset(9)
        obs[0,1]=env.spec.soc_min[0];obs[0,54]=0;self.assertLess(ess_bounds(env,obs[0])[1],1e-5)
        a,bw,cpu=decode(env,np.zeros((env.num_agents,1)));self.assertTrue((bw>0).all() and (cpu>0).all())
        if env.num_agents==18:
            _,idle_bw,idle_cpu=decode(env,np.full((env.num_agents,1),-20.))
            self.assertTrue((idle_bw==0).all() and (idle_cpu==0).all())
        public=env.encode()[0][0]
        if public[11]>0:self.assertLess(a[4]/(public[11]*1000),.03)

    def test_four_algorithms_batch_checkpoint_and_resume(self):
        from vpp_research.train import train,load,evaluate
        from vpp_research.ledger import audit_training
        for method in ('pareto','ordinary','central','fixed'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as d:
                c=Config.load('configs/research_smoke.json');c.research_version=4;c.episodes=3;c.horizon=2;c.rollout_episodes=2;out=Path(d)/method
                train(c,None,out,method=method,trace_path=out/'trajectory.jsonl');state,_,_=load(out/'latest.pt')
                self.assertEqual(state['observation_version'],'c3-80-public-command-queue-v4')
                audit=audit_training(out);self.assertTrue(audit['consistent']);self.assertEqual(audit['runs'][0]['successful_calls'],6)
                train(c,None,out,method=method,trace_path=out/'trajectory.jsonl',resume=True)
                self.assertEqual(audit_training(out)['runs'][0]['successful_calls'],6)
                self.assertFalse(evaluate(out/'latest.pt',[[.2,.3,.5]],[99999])[0]['failed'])

    def test_interrupted_batch_replayed_and_charged(self):
        from unittest.mock import patch
        from vpp_research.train import train
        from vpp_research.ledger import audit_training
        original=ParetoV4.update;calls=0
        def interrupted(agent,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('验收：第二次更新中断')
            return original(agent,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            c=Config.load('configs/research_smoke.json');c.research_version=4;c.horizon=2;c.episodes=3;c.rollout_episodes=2;out=Path(d)/'interrupted'
            with patch.object(ParetoV4,'update',interrupted),self.assertRaisesRegex(RuntimeError,'更新中断'):train(c,None,out,trace_path=out/'trajectory.jsonl')
            train(c,None,out,trace_path=out/'trajectory.jsonl',resume=True);item=audit_training(out)['runs'][0]
            self.assertEqual(item['committed_steps'],6);self.assertEqual(item['successful_calls'],8);self.assertTrue(item['consistent'])

if __name__=='__main__':unittest.main()
