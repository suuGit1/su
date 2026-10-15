"""检查actor剩余控制权、独立资源动作、回退和真实队列服务。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.cyber import CyberSpec,CyberPipeline
from vpp_mappo.coordinator_projection import PublicServiceProjection

class ProjectionTests(unittest.TestCase):
    def make(self,B=10000,C=1e7):
        c=Config(coordinator_revision='public_service',coordinator_mode='schedule',risk_adaptive=True,dt_hours=1,command_timing=True,control_cycles=1e9,coordinator_active_channels=(1,0,1,0,1,1))
        spec=CyberSpec(bandwidth_bps=B,cpu_cycles_per_second=C)
        q=PublicServiceProjection(c,SimpleNamespace(soc_min=[.1],soc_max=[.9]),SimpleNamespace(ev_station_kw=100,dr_backlog_kwh=0,dr_repay_kw=0),spec)
        x=np.zeros(54);x[1]=.5;return q,x,spec
    def test_actor_authority(self):
        q,x,s=self.make();a=np.array([9,0,1,0,1,1]);b=np.array([1,0,9,0,1,1]);cpu=np.ones(6)
        first=q.allocate(x,a,cpu);second=q.allocate(x,b,cpu)
        self.assertTrue(first[2]['projection_applied']);self.assertGreater(np.linalg.norm(first[0]-second[0]),.1)
        np.testing.assert_allclose(first[1],second[1])
        third=q.allocate(x,a,b);self.assertGreater(np.linalg.norm(first[1]-third[1]),.1)
    def test_pipeline_completion(self):
        for B,C in [(10000,1e7),(2000,1e7),(10000,2e6)]:
            q,x,s=self.make(B,C);bw,cpu,r=q.allocate(x,[100,0,0,0,0,0],[0,0,0,0,0,100])
            self.assertTrue(r['projection_applied']);pipe=CyberPipeline(replace(s,cpu_cycles_per_second=C-1e9/3600),42)
            pipe.enqueue([dict(v=1)]*6,bw>0);jobs,_,_=pipe.advance(3600,bw,cpu)
            self.assertEqual({j['channel'] for j in jobs},{0,2,4,5})
            self.assertLessEqual(sum(bw),1+1e-12);self.assertLessEqual(sum(cpu),1+1e-12)
    def test_fallback(self):
        q,x,s=self.make(2000,2e6);a=np.ones(6);bw,cpu,r=q.allocate(x,a,a)
        self.assertFalse(r['projection_applied']);self.assertEqual(bw[1]+bw[3]+cpu[1]+cpu[3],0)
    def test_no_hidden_queue_interface(self):
        q,x,s=self.make();x[24:30]=[.1,0,.2,0,.3,.4];x[30:36]=.1
        bw,cpu,r=q.allocate(x,np.ones(6),np.ones(6));self.assertNotIn('service_state',r)
if __name__=='__main__':unittest.main()
