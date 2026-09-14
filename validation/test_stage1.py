"""直接加载仿真核心，隔离旧版包入口对 PyTorch 和 Web 服务的强制导入。"""
import sys
import types
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('stage1_core')
pkg.__path__ = [str(root / 'src/vpp/dt_c3')]
sys.modules['stage1_core'] = pkg
from stage1_core.safety import SafetyLimits, SafetyShield
from stage1_core.environment import PowerIoTVPPEnvironment, VPPEnvConfig
from stage1_core.experiment_results import AdvancedExperimentResult


class IntegrityTests(unittest.TestCase):
    def env(self, **kwargs):
        return PowerIoTVPPEnvironment(VPPEnvConfig(horizon_steps=2, use_grid_constraints=False, **kwargs))

    def test_discharge_boundary(self):
        e = self.env()
        a = e.shield.project({'ess_soc': 0.11, 'ev_soc': 0.5}, {'ess_power_kw': 250}).corrected_action
        self.assertAlmostEqual(a['ess_power_kw'], 19.0)
        self.assertAlmostEqual(e._next_soc(0.11, a['ess_power_kw'], 500), 0.1)

    def test_charge_boundary_custom_efficiency(self):
        e = self.env(dt_hours=0.5, safety_limits=SafetyLimits(charge_efficiency=0.8))
        a = e.shield.project({'ess_soc': 0.89, 'ev_soc': 0.5}, {'ess_power_kw': -250}).corrected_action
        self.assertAlmostEqual(e._next_soc(0.89, a['ess_power_kw'], 500), 0.9)

    def test_unshielded_violation_is_measured(self):
        e = self.env()
        e.true_state['ess_soc'] = 0.11
        r = e.step({'ess_power_kw': 250}, apply_safety=False)
        self.assertIn('ess_soc_min', r.info['physical_violations'])
        self.assertLess(r.info['post_state']['ess_soc'], 0.1)
        self.assertGreater(r.info['constraint_violations'], 0)

    def test_grid_settlement_uses_true_load(self):
        e = self.env()
        before = dict(e.true_state)
        r = e.step({'grid_power_kw': -800}, apply_safety=False)
        expected = before['load_kw'] - before['pv_kw'] - before['wind_kw']
        self.assertAlmostEqual(r.info['grid_power_kw'], expected)
        self.assertAlmostEqual(r.info['grid_deviation_kw'], expected + 800)

    def test_grid_limit_is_measured(self):
        e = self.env(safety_limits=SafetyLimits(grid_import_max_kw=1))
        r = e.step({}, apply_safety=False)
        self.assertIn('grid_import', r.info['physical_violations'])

    def test_observe_is_idempotent(self):
        e = self.env()
        n = len(e.channel.history())
        first = e.observe()
        first['ess_soc'] = -123
        e.observe(False)
        self.assertEqual(len(e.channel.history()), n)
        self.assertNotEqual(e.observe()['ess_soc'], -123)
        e.step({})
        self.assertEqual(len(e.channel.history()), n + 5)

    def test_communication_billed_once(self):
        e = self.env()
        costs = [e.step({}).info['communication_cost'] for _ in range(2)]
        self.assertAlmostEqual(sum(costs), e.channel.total_communication_cost())

    def test_curtailment_increases_import_and_matches_grid(self):
        e = self.env()
        before = dict(e.true_state)
        r = e.step({'curtailment_kw': 10}, apply_safety=False)
        self.assertAlmostEqual(r.info['grid_power_kw'], before['load_kw'] - before['pv_kw'] - before['wind_kw'] + 10)
        injections = e.grid.vpp_to_bus_injections(before, r.info['applied_action'])
        self.assertAlmostEqual(sum(injections.values()) + r.info['grid_power_kw'], 0)

    def test_terminal_state_and_reset(self):
        e = self.env()
        e.step({'ess_power_kw': 20}, apply_safety=False)
        r = e.step({'ess_power_kw': 20}, apply_safety=False)
        self.assertTrue(r.done)
        self.assertEqual(e.true_state['ess_soc'], r.info['post_state']['ess_soc'])
        with self.assertRaises(RuntimeError):
            e.step({})
        e.reset()
        self.assertAlmostEqual(e.true_state['ess_soc'], 0.55)

    def test_ablation_groups_stay_separate(self):
        result = AdvancedExperimentResult()
        for variant, cost in [('full', 10), ('no_dt', 20)]:
            for seed in (1, 2):
                result.add_metrics(dict(scenario='ablation', method='mappo', variant=variant, seed=seed, total_cost=cost))
        result.build_summary()
        self.assertEqual(len(result.summary_rows), 2)
        self.assertEqual({r['variant']: r['total_cost_mean'] for r in result.summary_rows}, {'full': 10, 'no_dt': 20})
        self.assertTrue(all(r['n_seeds'] == 2 for r in result.summary_rows))

    def test_nonfinite_action_rejected(self):
        with self.assertRaises(ValueError):
            self.env().step({'ess_power_kw': float('nan')})


if __name__ == '__main__':
    unittest.main(verbosity=2)
