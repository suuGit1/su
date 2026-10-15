"""使用真实队列推进验证服务上界，而非只重复分配公式。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.cyber import CyberSpec,CyberPipeline
from vpp_mappo.coordinator_service import ServiceCoordinator

class ServiceTests(unittest.TestCase):
    def scenario(self,B,C):
        spec=CyberSpec(bandwidth_bps=B,cpu_cycles_per_second=C)
        pipe=CyberPipeline(spec,42);c=Config(dt_hours=1,coordinator_mode='schedule',risk_adaptive=True)
        q=ServiceCoordinator(c,SimpleNamespace(soc_min=[.1],soc_max=[.9]),SimpleNamespace(ev_station_kw=100,dr_backlog_kwh=100,dr_repay_kw=50),[1,0,1,0,1,1])
        q.service_state=dict(tx_bits=[0.]*6,cpu_cycles=[0.]*6,tx_count=[0]*6,cpu_count=[0]*6,queue_capacity=4,
                            deadline_seconds=3600.,bandwidth_bps=B,cpu_cycles_per_second=C,packet_overhead_bits=1e6,payload_bound_bits=8192.,update_cycles=1e9)
        x=np.zeros(54);x[1]=.5;b,p,r=q.allocate(x,np.ones(6),np.ones(6))
        return pipe,b,p,r
    def test_completion_using_pipeline(self):
        for B,C in [(10000.,1e7),(2000.,1e7),(10000.,2e6),(2000.,2e6)]:
            pipe,b,p,r=self.scenario(B,C);pipe.enqueue([dict(v=1)]*6,b>0);done,_,_=pipe.advance(3600,b,p)
            self.assertTrue(set(r['fresh_admitted'])<=set(x['channel'] for x in done))
            self.assertLessEqual(sum(b),1+1e-12);self.assertLessEqual(sum(p),1+1e-12)
            self.assertEqual(b[1]+b[3]+p[1]+p[3],0)
    def test_zero_capacity(self):
        pipe,b,p,r=self.scenario(0,0);self.assertEqual(sum(b)+sum(p),0);self.assertEqual(r['selected_channels'],[])
    def test_insufficient_single_job(self):
        pipe,b,p,r=self.scenario(1,1);self.assertEqual(r['selected_channels'],[]);self.assertIsNone(r['deterministic_completion_bound_seconds'])
if __name__=='__main__':unittest.main()
