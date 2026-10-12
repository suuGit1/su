"""检查无效动作梯度、偏好退化、PopArt输出守恒和回合边界。"""
import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vpp_research.pareto_v44 import ParetoV44,balanced_preference
from vpp_research.pareto_agent import vector_gae
class Tests(unittest.TestCase):
 def setUp(self):torch.manual_seed(3);torch.set_num_threads(1)
 def model(self,**kw):return ParetoV44(80,18,hidden=16,action_mask=[1,0,1,0,1,1]*3,**kw)
 def test_fixed_and_sampling(self):
  m=self.model(fixed_preference=[1,0,0]);o=torch.randn(18,80)
  self.assertTrue(torch.equal(m.distribution(o,torch.tensor([1.,0,0])).mean,m.distribution(o,torch.tensor([0.,1,0])).mean))
  rng=np.random.default_rng(3);self.assertTrue(np.array_equal(np.array([balanced_preference(rng,i) for i in range(3)]),np.eye(3)))
 def test_popart_preserves_values(self):
  m=self.model();o=torch.randn(8,18,80);w=torch.ones(8,3)/3;before=m.value(o,w).detach()
  m.update_statistics(torch.randn(8,3)*torch.tensor([10.,100.,1.])+torch.tensor([20.,-90.,2.]))
  torch.testing.assert_close(m.value(o,w),before,atol=2e-5,rtol=1e-4)
 def test_inactive_action_no_gradient_effect(self):
  m=self.model();other=copy.deepcopy(m);rng=np.random.default_rng(4)
  roll={k:[] for k in ('obs','preferences','actions','logprobs','values','vectors','dones')}
  for i in range(8):
   o=rng.normal(size=(18,80));w=balanced_preference(rng,i//2);a,lp,v=m.act(o,w)
   for k,x in zip(roll,(o,w,a,lp,v,rng.normal(size=3),i%2==1)):roll[k].append(x)
  altered=copy.deepcopy(roll)
  for a in altered['actions']:a[np.asarray(m.action_mask)==0]+=3
  m.update(roll,epochs=2);other.update(altered,epochs=2)
  for a,b in zip(m.parameters(),other.parameters()):torch.testing.assert_close(a,b)
 def test_critic_budget(self):
  m=self.model(critic_gain=1.,critic_warmup=4,critic_extra=1);rng=np.random.default_rng(3)
  def rollout():
   r={k:[] for k in ('obs','preferences','actions','logprobs','values','vectors','dones')}
   for i in range(4):
    o=rng.normal(size=(18,80));w=np.ones(3)/3;a,lp,v=m.act(o,w)
    for k,x in zip(r,(o,w,a,lp,v,rng.normal(size=3),i==3)):r[k].append(x)
   return r
  self.assertEqual(m.update(rollout(),epochs=2)['critic_updates'],4)
  self.assertEqual(m.update(rollout(),epochs=2)['critic_updates'],3)
 def test_gae_terminal_no_leak(self):
  a,_=vector_gae([[1,2,3],[100,200,300]],np.zeros((3,3)),[True,True])
  np.testing.assert_allclose(a[0],[1,2,3])
if __name__=='__main__':unittest.main()
