"""真实数据入口禁止合成回退、测试日期重复和资源协议混用。"""
import argparse
import tempfile
import unittest
from vpp_research.real_inputs import connect
from vpp_research.real_campaign import parse_seeds


class RealInputTests(unittest.TestCase):
    def test_default_bundle_uses_real_days_and_disables_unavailable_resources(self):
        c,paths,sets,p=connect('data/real')
        self.assertEqual([len(sets[k].profiles) for k in ('train','validation','test')],[31,27,29])
        self.assertIsNone(c.synthetic_carbon_g_per_kwh)
        self.assertTrue(all(not s for s in c._ev_bundle['scenarios'].values()))
        self.assertEqual(c._flex_record['dr_shift_kw'],0)
        self.assertEqual(c._flex_record['dr_shed_kw'],0)
        self.assertEqual(c.horizon,24);self.assertEqual(c.dt_hours,1)

    def test_missing_data_never_falls_back_to_synthetic(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(FileNotFoundError):connect(d)

    def test_sce_option_is_explicit_cross_region_and_training_only(self):
        c,_,_,p=connect('data/real',dr_mode='sce-derated')
        self.assertEqual(c._flex_record['dr_shed_kw'],0)
        self.assertFalse(p['dr']['certified'])
        self.assertEqual(c._flex_record['dr_shift_kw'],0)

    def test_seed_list_is_open_but_unique(self):
        self.assertEqual(parse_seeds('1,2,103'),[1,2,103])
        for value in ('1,1','-1','abc'):
            with self.assertRaises(argparse.ArgumentTypeError):parse_seeds(value)

if __name__=='__main__':unittest.main()
