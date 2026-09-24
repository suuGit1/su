import json
import tempfile
import unittest
from pathlib import Path
from vpp_research.real_benchmarks import budgets
from vpp_research.real_report import report


class BenchmarkTests(unittest.TestCase):
    def test_total_budget_includes_all_members(self):
        self.assertEqual(budgets(10,3),[4,3,3])
        self.assertEqual(sum(budgets(101,6)),101)
        with self.assertRaises(ValueError):budgets(2,3)

    def test_unconfirmed_reserve_excluded_and_reference_validation_only(self):
        good=dict(failed=False,preference=[1,0,0],vector=[-1,-2,3],violations=0,ac_violations=0,
                  ac_failed=0,reserve_invalid=0,ev_unmet_kwh=0,dr_backlog_kwh=0)
        bad={**good,'reserve_invalid':1,'vector':[-.1,-.1,100]}
        entry=dict(seed=1,method='ordinary',results=[bad],validation=[good])
        with tempfile.TemporaryDirectory() as d:
            Path(d,'results.json').write_text(json.dumps(dict(manifest=dict(eval_days=1,selection_days=1),entries=[entry])))
            result=report(d)
            self.assertEqual(result['empirical_validation_reference'],[good['vector']])
            self.assertEqual(result['rows'][0]['hv'],0)
            self.assertIsNone(result['rows'][0]['igd'])
            self.assertTrue(Path(d,'summary.csv').exists())
