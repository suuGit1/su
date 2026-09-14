"""新训练入口的行为验证，包含真实官方 PPO 更新与检查点往返。"""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vpp_mappo.config import Config
from vpp_mappo.environment import VPPAdapter
from vpp_mappo.algorithms import OfficialPPO, to_numpy
from vpp_mappo.runner import train, evaluate, seed_all
from vpp_mappo.data import CSVProfiles


class Stage2Tests(unittest.TestCase):
    def test_local_critic_is_distinct(self):
        cfg = Config(horizon=4, episodes=1, hidden_size=16)
        e = VPPAdapter(cfg)
        obs, share = e.reset(1)
        m = OfficialPPO(cfg, e)
        cfg.algorithm = 'ippo'
        i = OfficialPPO(cfg, e)
        self.assertEqual(m.buffer().share_obs.shape[-1], 30)
        self.assertEqual(i.buffer().share_obs.shape[-1], 10)
        self.assertEqual(m.critic_obs(obs, share).shape, (3, 30))
        self.assertEqual(i.critic_obs(obs, share).shape, (3, 10))

    def test_actor_private_state_is_local(self):
        e = VPPAdapter(Config(horizon=4))
        e.reset(1)
        state = e.env.observe(False)
        a, _ = e.encode(state)
        state['ev_soc'] += 0.1
        b, _ = e.encode(state)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(a[1], b[1]))

    def test_raw_action_log_probability_is_consistent(self):
        cfg = Config(horizon=4, hidden_size=16)
        seed_all(7, 1)
        e = VPPAdapter(cfg)
        obs, share = e.reset(1)
        a = OfficialPPO(cfg, e)
        a.trainer.prep_rollout()
        rnn = np.zeros((3, 1, 16), np.float32)
        masks = np.ones((3, 1), np.float32)
        with torch.no_grad():
            _, action, logp, _, _ = a.policy.get_actions(share, obs, rnn, rnn, masks)
            _, new_logp, _ = a.policy.evaluate_actions(share, obs, rnn, rnn, action, masks)
        np.testing.assert_allclose(to_numpy(logp), to_numpy(new_logp), atol=1e-6)
        raw = np.array([[100.0], [-100.0], [0]], np.float32)
        e.step(raw)
        np.testing.assert_array_equal(raw, [[100.0], [-100.0], [0]])

    def test_true_terminal_masks_bootstrap(self):
        cfg = Config(horizon=2, hidden_size=16)
        a = OfficialPPO(cfg, VPPAdapter(cfg))
        b = a.buffer()
        b._use_valuenorm = False
        b.rewards[:] = 1
        b.masks[-1] = 0
        b.compute_returns(np.full((1, 3, 1), 999, np.float32))
        np.testing.assert_allclose(b.returns[-2], 1)

    def test_train_checkpoint_eval_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = Config(episodes=2, horizon=4, hidden_size=16, ppo_epoch=2)
            history = train(cfg, root / 'train')
            self.assertTrue(all(r['actor_parameter_delta'] > 0 for r in history))
            a = evaluate(root / 'train/latest.pt', root / 'eval1', episodes=2)
            b = evaluate(root / 'train/latest.pt', root / 'eval2', episodes=2)
            self.assertEqual(a, b)
            with self.assertRaises(ValueError):
                evaluate(root / 'train/latest.pt', root / 'overlap', seed=1)
            with self.assertRaises(ValueError):
                train(cfg, root / 'train')

    def test_csv_validation_and_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'profiles.csv'
            with p.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['scenario', 'step', 'load_kw', 'pv_kw', 'wind_kw', 'price'])
                for step in range(4):
                    w.writerow(['day1', step, 500, 100, 50, 0.2])
            cfg = Config(horizon=4, episodes=1, hidden_size=16, ppo_epoch=1, train_csv=str(p))
            e = VPPAdapter(cfg, str(p))
            e.reset(1)
            self.assertEqual(e.env.true_state['load_kw'], 500)
            history = train(cfg, Path(temp) / 'train')
            self.assertEqual(len(history), 1)
            duplicate = Path(temp) / 'duplicate.csv'
            duplicate.write_bytes(p.read_bytes())
            with self.assertRaises(ValueError):
                evaluate(Path(temp) / 'train/latest.pt', Path(temp) / 'eval', episodes=1, csv_path=str(duplicate))
            with self.assertRaises(ValueError):
                CSVProfiles(p, 5)

    def test_bad_config_fails(self):
        for cfg in [Config(horizon=1), Config(lr=-1), Config(algorithm='iac'), Config(safety='false')]:
            with self.assertRaises(ValueError):
                cfg.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
