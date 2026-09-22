"""验证双算法断点恢复与集成日志，不用短训练结果推断性能优势。"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from vpp_mappo.config import Config
from vpp_research.train import train


class ResumeTests(unittest.TestCase):
    def test_both_algorithms_match_uninterrupted_episode_updates(self):
        c=replace(Config.load('configs/integrated_ieee33.json'),horizon=2,episodes=2)
        def equal(a,b):
            if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        for method in ('ordinary','pareto'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as folder:
                out=Path(folder)
                full=train(c,None,out/'full',method=method)
                train(replace(c,episodes=1),None,out/'resume',method=method)
                resumed=train(c,None,out/'resume',method=method,resume=True)
                equal(full['agent'],resumed['agent'])
                self.assertEqual(full['training_preferences'],resumed['training_preferences'])
                with self.assertRaises(ValueError):
                    train(replace(c,episodes=3,lr=.001),None,out/'resume',method=method,resume=True)

if __name__=='__main__':unittest.main()
