"""区间尺度的训练隔离、日期块校准、兼容性及公开观察接口。"""
import copy
import unittest
import numpy as np
from vpp_research.dt import fit,predict,interval_halfwidth,assess,TARGETS


def records(start,days):
    rng=np.random.default_rng(start);rows=[]
    for d in range(start,start+days):
        for t in range(4):
            x=rng.uniform(.1,.8,54);x[0]=t/4
            y=x[TARGETS]+rng.normal(0,.01+(.02*t),9)
            rows.append(dict(scenario=str(d),step=t,x=x.tolist(),y=y.tolist()))
    return rows


class AdaptiveDTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=records(0,10);cls.calibration=records(30,12);cls.test=records(60,4)
        cls.model=fit(cls.training,cls.calibration,interval_mode='adaptive')

    def test_calibration_does_not_fit_prediction_or_scale(self):
        changed=copy.deepcopy(self.calibration)
        for row in changed:row['y']=(np.array(row['y'])+.5).tolist()
        other=fit(self.training,changed,interval_mode='adaptive')
        for key in ('coef','mean','scale','scale_coef','output_scale'):
            np.testing.assert_array_equal(other[key],self.model[key])
        self.assertNotEqual(other['calibration_quantile'],self.model['calibration_quantile'])

    def test_same_point_prediction_as_constant(self):
        constant=fit(self.training,self.calibration)
        x=np.array([r['x'] for r in self.test])
        np.testing.assert_array_equal(predict(constant,x)[0],predict(self.model,x)[0])
        widths=interval_halfwidth(self.model,x)
        self.assertEqual(widths.shape,(16,9));self.assertTrue(np.isfinite(widths).all());self.assertTrue((widths>=0).all())
        self.assertGreater(float(widths.std(axis=0).max()),0)
        np.testing.assert_array_equal(interval_halfwidth(constant,x)[0],constant['halfwidth'])

    def test_rank_and_leakage(self):
        self.assertEqual(self.model['calibration_rank'],12)
        with self.assertRaises(ValueError):fit(self.training,self.training)
        with self.assertRaises(ValueError):fit(self.training,records(30,8))
        with self.assertRaises(ValueError):assess(self.model,self.calibration)
        with self.assertRaises(ValueError):assess(self.model,[])
        with self.assertRaises(ValueError):fit(records(0,4),self.calibration,interval_mode='adaptive')

    def test_public_observation_width_before_point_replacement(self):
        from vpp_research.environment import ResearchEnv
        from vpp_mappo.config import Config
        c=Config.load('configs/research_smoke.json')
        e=ResearchEnv(c,dt_model=self.model,vector_metrics=False)
        obs,_=e.reset(42);raw=e.raw_observation()
        np.testing.assert_allclose(obs[0,54:63],interval_halfwidth(self.model,raw),rtol=1e-6)

    def test_assessment_uses_actual_widths(self):
        x=np.array([r['x'] for r in self.test]);a=assess(self.model,self.test)
        self.assertAlmostEqual(a['mean_interval_width'],2*interval_halfwidth(self.model,x).mean())
        self.assertEqual(a['scenarios'],4)

if __name__=='__main__':unittest.main()
