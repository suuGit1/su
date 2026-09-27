"""验证事务恢复、缺步拒绝、轨迹复算及向量偏好接口。"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from vpp_research.ledger import Ledger
from vpp_research.evidence_v3 import audit_trace
from vpp_research.pareto_agent import vector_gae,ParetoAgent


class V3Tests(unittest.TestCase):
    def test_unknown_call_blocks_resume(self):
        with tempfile.TemporaryDirectory() as d:
            x=Ledger(Path(d)/'calls.sqlite');x.begin('a',0,0);x.close()
            x=Ledger(Path(d)/'calls.sqlite')
            with self.assertRaises(ValueError):x.reconcile([],2)
            self.assertEqual(x.summary()['unknown_calls'],1);x.close()

    def test_retries_are_charged_but_only_checkpoint_committed(self):
        with tempfile.TemporaryDirectory() as d:
            x=Ledger(Path(d)/'calls.sqlite')
            x.finish(x.begin('interrupted',0,0),'success')
            for t in range(2):x.finish(x.begin('resumed',0,t),'success')
            x.reconcile(['resumed'],2)
            self.assertEqual(x.summary()['successful_calls'],3)
            self.assertEqual(x.summary()['committed_steps'],2)
            x.reconcile([],2);self.assertEqual(x.summary()['committed_steps'],0);x.close()

    def test_missing_step_blocks_commit(self):
        with tempfile.TemporaryDirectory() as d:
            x=Ledger(Path(d)/'calls.sqlite');x.finish(x.begin('a',0,1),'success')
            with self.assertRaises(ValueError):x.commit_episode(0,'a',2)
            x.close()

    def test_trace_catches_missing_duplicate_and_wrong_vector(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trajectory.jsonl'
            info=dict(objective_vector=[-1,-2,3],constraint_violations=0,ac_violations=0,reserve_valid=True,
                      guard_certificate_survived=True,ev_unmet_kwh=0,dr_backlog_kwh=0)
            rows=[dict(step=t,feedback=info) for t in range(2)]
            ledger=Ledger(Path(d)/'evaluation.sqlite')
            for t in range(2):ledger.finish(ledger.begin('evaluation',0,t),'success')
            ledger.close()
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertTrue(audit_trace(p,dict(env_steps=2,vector=[-2,-4,6]))['consistent'])
            self.assertFalse(audit_trace(p,dict(env_steps=2,vector=[-2,-4,7]))['consistent'])
            rows[1]['step']=0;p.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertFalse(audit_trace(p)['consistent'])

    def test_vector_gae_keeps_objectives_and_terminal_mask(self):
        r=np.array([[1,10,100],[2,20,200]],float);v=np.zeros((3,3));v[-1]=999
        adv,ret=vector_gae(r,v,[False,True],gamma=1,lam=1)
        np.testing.assert_allclose(adv,[[3,30,300],[2,20,200]])
        np.testing.assert_allclose(ret,adv)

    def test_actor_preference_and_vector_critic_interfaces(self):
        import torch
        torch.manual_seed(7);agent=ParetoAgent(65,18)
        obs=np.zeros((18,65),np.float32)
        a,_,v=agent.act(obs,[1,0,0],True);b,_,_=agent.act(obs,[0,1,0],True)
        self.assertEqual(v.shape,(3,));self.assertGreater(float(np.linalg.norm(a-b)),1e-6)
