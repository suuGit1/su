"""统一预算中自适应选择成本不可遗漏。"""
import unittest
from vpp_research.fair_budget import allocation

class BudgetTests(unittest.TestCase):
    def test_all_methods_have_same_training_plus_selection_budget(self):
        expected={'ordinary':(20,4),'fixed':(12,12),'pareto':(8,16),'ols':(8,16)}
        for method,(train,selection) in expected.items():
            b=allocation(method,12,2)
            self.assertEqual((b['training_steps'],b['selection_steps']),(train,selection))
            self.assertEqual(train+selection,24)
    def test_invalid_split_is_rejected_before_training(self):
        with self.assertRaises(ValueError):allocation('ols',10,8)
        with self.assertRaises(ValueError):allocation('pareto',8,8)



    def test_early_stopped_ols_keeps_actual_spend_and_is_not_a_matched_result(self):
        import tempfile
        from unittest.mock import patch
        from vpp_mappo.config import Config
        from vpp_research.fair_budget import run
        c=Config.load('configs/fair_budget_smoke.json')
        state={'status':'no_unvisited_corner; unused_budget_not_spent',
               'completed_training_steps':2,'completed_selection_steps':4}
        with tempfile.TemporaryDirectory() as d, patch('vpp_research.fair_budget.train_ols',return_value=state):
            result=run(c,None,d,families=('ols',))['entries'][0]
        self.assertTrue(result['failed']);self.assertFalse(result['budget_matched'])
        self.assertEqual(result['training_steps']+result['selection_steps'],6)
        self.assertEqual(result['test_steps'],0)

if __name__=='__main__':unittest.main()
