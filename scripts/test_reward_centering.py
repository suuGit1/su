"""验证外生基准的策略无关性与原目标可恢复，不声称控制收益。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vpp_research.reward_centering import reference_vector
class Tests(unittest.TestCase):
 def test_fixed_horizon_return_difference(self):
  rng=np.random.default_rng(8);base=[]
  for t in range(24):
   base.append(reference_vector(dict(load_kw=1000+10*t,pv_kw=200,wind_kw=100,price=.1,carbon_g_per_kwh=200),.8,1,[100]*3))
  b=np.asarray(base);r1=rng.normal(size=(24,3));r2=rng.normal(size=(24,3));discount=.99**np.arange(24);w=np.array([.2,.3,.5])
  before=(discount[:,None]*(r1-r2)).sum(0)@w
  after=(discount[:,None]*((r1-b)-(r2-b))).sum(0)@w
  self.assertAlmostEqual(before,after,places=12)
  np.testing.assert_allclose((r1-b)+b,r1,atol=1e-14)
 def test_export_and_negative_price(self):
  row=dict(load_kw=100,pv_kw=200,wind_kw=100,price=-.1,carbon_g_per_kwh=200)
  b=reference_vector(row,.8,1,[100]*3)
  np.testing.assert_allclose(b,[-.16,0,0])
if __name__=='__main__':unittest.main()
