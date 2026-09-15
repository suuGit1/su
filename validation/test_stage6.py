"""验证事件时序、预算守恒、延迟重放及策略/价值函数信息隔离。"""
import copy
import unittest
import numpy as np
from vpp_mappo.cyber import CyberSpec,CyberPipeline


class PipelineTests(unittest.TestCase):
    def pipeline(self):
        return CyberPipeline(CyberSpec(bandwidth_bps=10,cpu_cycles_per_second=20,packet_overhead_bits=0,update_cycles=100),1)
    def test_transmission_then_computation(self):
        p=self.pipeline();events=p.enqueue([{'v':1}]*6,[True,False,False,False,False,False]);bits=events[0]['bits']
        done,events,used=p.advance(bits/10,[1,0,0,0,0,0],[1,0,0,0,0,0])
        self.assertEqual(done,[]);self.assertEqual(used['cpu_cycles'],0)
        done,events,used=p.advance(5,[1,0,0,0,0,0],[1,0,0,0,0,0])
        self.assertEqual(len(done),1);self.assertAlmostEqual(done[0]['finished'],bits/10+5)
    def test_zero_cpu_hides_payload(self):
        p=self.pipeline();p.enqueue([{'v':1}]*6,[True]*6)
        done,_,_=p.advance(100,np.ones(6),np.zeros(6));self.assertEqual(done,[]);self.assertEqual(sum(map(len,p.cpu)),6)
        done,_,_=p.advance(30,np.ones(6),np.ones(6));self.assertEqual(len(done),6)
    def test_budget_and_carryover(self):
        p=self.pipeline();p.enqueue([{'v':1}]*6,[True]*6)
        _,_,used=p.advance(1,np.ones(6),np.ones(6))
        self.assertAlmostEqual(used['tx_bits'],10);self.assertLessEqual(used['cpu_cycles'],20)
        self.assertEqual(sum(map(len,p.tx)),6)
    def test_overflow_and_loss(self):
        p=self.pipeline();p.spec.queue_capacity=1;p.spec.packet_loss=1
        p.enqueue([{}]*6,[True]*6);events=p.enqueue([{}]*6,[True]*6)
        self.assertEqual(sum(e['kind']=='tx_overflow' for e in events),6)
        done,events,_=p.advance(100,np.ones(6),np.ones(6));self.assertEqual(done,[])
        self.assertEqual(sum(e['kind']=='packet_loss' for e in events),6)
    def test_seed_reproducible(self):
        a=self.pipeline();b=self.pipeline();a.spec.packet_loss=b.spec.packet_loss=.5
        for p in (a,b):p.enqueue([{}]*6,[True]*6)
        self.assertEqual(a.advance(100,np.ones(6),np.ones(6)),b.advance(100,np.ones(6),np.ones(6)))


class ClosedLoopTests(unittest.TestCase):
    def env(self,**overrides):
        from vpp_mappo.config import Config
        from vpp_mappo.environment import VPPAdapter
        c=Config.load('configs/sessions_smoke.json');c.cyber_mode='joint'
        c._cyber_record=dict(bandwidth_bps=1e8,cpu_cycles_per_second=1e12,**overrides)
        e=VPPAdapter(c);e.reset(1);return e
    def test_unreceived_truth_not_in_actor_or_critic(self):
        a=self.env();b=self.env();b.core.soc[0]=.7
        b.core.profiles['wind_kw'][0]+=50;b.core.remaining['demo_0']+=1
        for x,y in zip(a.encode(),b.encode()):np.testing.assert_array_equal(x,y)
    def test_basic_dt_replays_after_delayed_sample(self):
        e=self.env();dt=e.dt
        for _ in range(3):dt.command([10,0,0,0,0,0])
        self.assertTrue(dt.receive(dict(channel=0,sampled=900,payload={'soc':.6})))
        self.assertFalse(dt.receive(dict(channel=0,sampled=0,payload={'soc':.1})))
        expected=.6-2*10*.25/(e.spec.efficiency*e.spec.capacities[0])
        self.assertAlmostEqual(dt.estimate(3)[0]['soc'],expected)
    def test_hold_ablation(self):
        e=self.env();old=e.dt.cache[0]['soc'];e.dt.mode='hold';e.dt.command([100,0,0,0,0,0])
        self.assertEqual(e.dt.estimate(1)[0]['soc'],old)
    def test_new_ev_hidden_until_computed(self):
        e=self.env();a=np.ones((18,1));a[12:]=-10
        for _ in range(5):e.step(a)
        self.assertEqual(len(e.dt.estimate(e.t)[1]['sessions']),1)
        self.assertEqual(len(e.core.active()),2)
    def test_cpu_and_upload_change_observation(self):
        a=self.env();b=self.env();c=self.env()
        normal=np.ones((18,1));no_cpu=normal.copy();no_cpu[12:]=-10
        no_upload=normal.copy();no_upload[6:12]=-10
        for _ in range(3):
            oa,*_=a.step(normal);ob,*_=b.step(no_cpu);oc,*_=c.step(no_upload)
        self.assertFalse(np.array_equal(oa,ob));self.assertFalse(np.array_equal(oa,oc))
        self.assertGreater(b.dt.age(b.pipeline.now).mean(),a.dt.age(a.pipeline.now).mean())
    def test_checkpoint_restores_cyber_snapshot(self):
        import tempfile
        from pathlib import Path
        import torch
        from vpp_mappo.runner import train,evaluate
        e=self.env();cfg=e.config;cfg.episodes=1;cfg.horizon=2;cfg.hidden_size=16;cfg.ppo_epoch=1
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);train(cfg,root/'train')
            state=torch.load(root/'train/latest.pt',weights_only=True)
            state['config']['cyber_spec']='missing-cyber-parameters.json'
            torch.save(state,root/'snapshot.pt')
            rows=evaluate(root/'snapshot.pt',root/'eval',episodes=1)
            self.assertIn('mean_aoi_seconds',rows[0])
            self.assertEqual(state['cyber_spec']['bandwidth_bps'],1e8)

    def test_all_energy_constraints_still_checked(self):
        e=self.env();rows=[]
        for _ in range(e.config.horizon):
            *_,info=e.step(np.ones((18,1)));rows.append(info)
            self.assertEqual(info['constraint_violations'],0)
        self.assertAlmostEqual(sum(r['ev_unmet_kwh'] for r in rows),0)
        self.assertAlmostEqual(rows[-1]['dr_backlog_kwh'],0)

if __name__=='__main__':unittest.main()
