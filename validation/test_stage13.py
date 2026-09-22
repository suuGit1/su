"""当前状态内的备用目标复用不能跨状态泄漏。"""
import unittest
from unittest.mock import patch
from vpp_research.pcc_reserve import checked_capacity

class CapacityCacheTests(unittest.TestCase):
    def test_same_target_solved_once_but_next_confirmation_recomputes(self):
        calls=[]
        def solve(core,target):
            calls.append(target)
            return {},dict(pcc_target_kw=target,pcc_actual_kw=target,pcc_error_kw=0.)
        with patch('vpp_research.pcc_reserve.target_plan',side_effect=solve):
            first=checked_capacity(object(),100.,20.)
            self.assertEqual(len(calls),5)
            self.assertEqual(len(set(calls)),5)
            second=checked_capacity(object(),100.,20.)
            self.assertEqual(len(calls),10)
        self.assertEqual(first,second)
        self.assertEqual(len(first['samples']),5)

class Network69Tests(unittest.TestCase):
    def test_physical_units_and_net_injection_mapping(self):
        import numpy as np
        import pandapower as pp
        from vpp_mappo.dispatch import DispatchSpec
        from vpp_mappo.network import Network33
        from vpp_mappo.network_cases import case69
        n=case69()
        self.assertEqual((len(n.bus),len(n.line)),(69,68))
        self.assertAlmostEqual(n.load.p_mw.sum(),3.8021)
        self.assertAlmostEqual(n.load.q_mvar.sum(),2.6947)
        pp.runpp(n,algorithm='nr',numba=False,tolerance_mva=1e-10)
        expected=n.res_ext_grid.p_mw.sum()*1000
        grid=Network33(DispatchSpec.load('configs/dispatch_ieee69.json'))
        self.assertEqual(grid.paths.shape,(69,68))
        result=grid.audit(dict(load_kw=3802.1,pv_kw=0.,wind_kw=0.),np.zeros(3))
        self.assertTrue(result['ac_converged'])
        # 两种求解器的终止精度不同，允许 1 W 差异，远小于 PCC 的 100 W 容差。
        self.assertAlmostEqual(result['ac_grid_kw'],expected,delta=.001)
        # 原始负荷下的低电压必须报告，不能为通过测试而改变算例。
        self.assertLess(result['ac_min_vm_pu'],.95)
        self.assertGreater(result['ac_violations'],0)

    def test_network_mismatch_rejected(self):
        from vpp_mappo.config import Config
        from vpp_mappo.grid_environment import GridVPPAdapter
        c=Config.load('configs/research_ieee69.json')
        c.dispatch_spec='configs/dispatch_ieee33.json'
        with self.assertRaises(ValueError):GridVPPAdapter(c)

    def test_pcc_tracking_on_69_nodes(self):
        from vpp_mappo.config import Config
        from vpp_research.ac_safety import ACCheckedFlex
        from vpp_research.pcc_reserve import target_plan
        e=ACCheckedFlex(Config.load('configs/research_ieee69.json'),emergency=True)
        e.reset(19)
        p,_=e.plan()
        g=e.network.audit(e.row(),p[0]['action'])['ac_grid_kw']
        p,meta=target_plan(e,g+10)
        *_,info=e.step_physical(p)
        self.assertLessEqual(abs(info['ac_grid_kw']-g-10),.1)
        self.assertEqual(info['ac_violations'],0)

if __name__=='__main__':unittest.main()
