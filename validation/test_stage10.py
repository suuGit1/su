import unittest
import numpy as np
from vpp_mappo.config import Config
from vpp_research.environment import ResearchEnv
from vpp_research.parallel_env import VPPParallelEnv


class ParallelTests(unittest.TestCase):
    def test_full_trajectory_matches_existing_environment(self):
        c = Config.load('configs/research_smoke.json')
        wrapped = VPPParallelEnv(c)
        direct = ResearchEnv(c)
        obs, infos = wrapped.reset(seed=19)
        expected, _ = direct.reset(19)
        for a in wrapped.agents:
            self.assertIs(wrapped.reward_space(a), wrapped.reward_space(a))
            self.assertEqual(infos[a]['objective_version'], direct.objective_version)
        np.testing.assert_array_equal(wrapped.state(), expected.reshape(-1))
        for t in range(c.horizon):
            actions = np.full((direct.num_agents, 1), .1 * (t % 3), dtype=np.float32)
            obs, rewards, terminated, truncated, infos = wrapped.step(dict(zip(wrapped.possible_agents, actions)))
            expected, _, scalar, done, info = direct.step(actions)
            np.testing.assert_array_equal(wrapped.state(), expected.reshape(-1))
            for i, a in enumerate(wrapped.possible_agents):
                np.testing.assert_allclose(obs[a], expected[i])
                self.assertAlmostEqual(rewards[a][0], float(scalar[i, 0]), places=5)
                np.testing.assert_allclose(infos[a]['raw_objective_vector'], info['objective_vector'])
                self.assertEqual(terminated[a], done)
                self.assertFalse(truncated[a])
        self.assertFalse(wrapped.agents)
        with self.assertRaises(RuntimeError): wrapped.step({})
        wrapped.reset(seed=19)
        np.testing.assert_array_equal(wrapped.state(), direct.reset(19)[0].reshape(-1))

    def test_invalid_actions_do_not_advance_and_outputs_are_copies(self):
        e = VPPParallelEnv(Config.load('configs/research_smoke.json'))
        obs, _ = e.reset(seed=4)
        before = e.state()
        obs[e.agents[0]][:] = 99
        np.testing.assert_array_equal(e.state(), before)
        for actions in ({}, {a: [np.nan] for a in e.agents}, {a: [0, 0] for a in e.agents}):
            with self.assertRaises(ValueError): e.step(actions)
            self.assertEqual(e.env.t, 0)
        _, rewards, _, _, infos = e.step({a: [0.] for a in e.agents})
        a, b = e.possible_agents[:2]
        original = rewards[b].copy()
        rewards[a][:] = 99
        np.testing.assert_array_equal(rewards[b], original)
        infos[a]['raw_objective_vector'][0] = 99
        self.assertNotEqual(infos[b]['raw_objective_vector'][0], 99)




class ArchiveTests(unittest.TestCase):
    def test_incomplete_duplicate_and_infeasible_scenarios_are_not_selected(self):
        from vpp_research.pareto_archive import solutions
        def row(seed, w, vector, failed=False):
            return dict(seed=seed, preference=w, vector=vector, failed=failed,
                        violations=0, ac_violations=0, ac_failed=0, reserve_invalid=0,
                        ev_unmet_kwh=0, dr_backlog_kwh=0)
        rows = [row(s, [1,0,0], [1,1,1]) for s in [1,2]]
        rows += [row(s, [0,1,0], [2,2,2]) for s in [1,2]]
        rows += [row(s, [0,0,1], [99,99,99]) for s in [1,1]]
        rows += [row(1, [.5,.5,0], [99,99,99])]
        rows += [row(s, [.5,0,.5], None if s==2 else [99,99,99], s==2) for s in [1,2]]
        r = solutions(dict(seed=7, family='test', test=rows), [1,2])
        self.assertEqual([x['eligible'] for x in r], [True,True,False,False,False])
        self.assertEqual([x['non_dominated_within_policy_set'] for x in r], [False,True,False,False,False])


if __name__ == '__main__': unittest.main()
