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
    def test_transaction_payload_survives_projection_loss(self):
        from vpp_research.ledger import completed_evidence
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);x=Ledger(root/'interactions.sqlite')
            record=dict(attempt_id='a',episode=0,step=0,feedback=dict(actual=7.5))
            x.finish(x.begin('a',0,0),'success',record);x.close()
            x=Ledger(root/'interactions.sqlite');x.export();x.close()
            self.assertEqual(json.loads((root/'trajectory_complete.jsonl').read_text()),record)
            self.assertEqual(completed_evidence(root)['summary']['successful_calls'],1)
            p=root/'completed_evidence.json';r=json.loads(p.read_text());r['evidence']['calls'][0]['detail']['feedback']['actual']=9
            p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):completed_evidence(root)

    def test_partial_campaign_cannot_pass_evidence_gate(self):
        from unittest.mock import patch
        from vpp_research.evidence_v3 import audit_campaign
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan=dict(methods=['ordinary','pareto'],seeds=[1],eval_days=1,preferences=[[1,0,0]])
            entry=dict(method='ordinary',seed=1,results=[dict(trace_path='mock')],validation=[])
            data=dict(manifest=plan,entries=[entry],completed=True)
            (root/'results.json').write_text(json.dumps(data))
            with patch('vpp_research.evidence_v3.audit_trace',return_value=dict(consistent=True,errors=[])):
                self.assertFalse(audit_campaign(root)['consistent'])
                data['entries'].append(dict(entry,method='pareto'))
                (root/'results.json').write_text(json.dumps(data))
                self.assertTrue(audit_campaign(root)['consistent'])

    def test_completed_checkpoint_can_export_without_extra_interactions(self):
        from vpp_mappo.config import Config
        from vpp_research.train import train
        c=Config.load('configs/research_smoke.json');c.episodes=1;c.horizon=2
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';first=train(c,None,out,method='pareto')
            ledger=Ledger(out/'interactions.sqlite');before=ledger.summary();ledger.close()
            (out/'latest.pt').unlink()
            recovered=train(c,None,out,method='pareto',resume=True)
            ledger=Ledger(out/'interactions.sqlite');after=ledger.summary();ledger.close()
            self.assertEqual(before,after);self.assertEqual(first['training_preferences'],recovered['training_preferences'])
            self.assertTrue((out/'latest.pt').exists())

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
