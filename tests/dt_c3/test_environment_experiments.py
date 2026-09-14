from vpp.dt_c3 import DTC3SafeMARL, DTC3ExperimentSuite, ExperimentSuiteConfig, PowerIoTVPPEnvironment, VPPEnvConfig
from vpp.dt_c3.communication import CommunicationConfig


def test_environment_runs_one_step():
    env = PowerIoTVPPEnvironment(VPPEnvConfig(horizon_steps=4, communication=CommunicationConfig(max_delay_steps=1, fixed_delay_steps=1)))
    state = env.reset()
    method = DTC3SafeMARL()
    result = env.step(method.act(state))
    assert "cost" in result.info
    assert result.info["decision_time_s"] >= 0


def test_experiment_suite_quick_run():
    suite = DTC3ExperimentSuite(ExperimentSuiteConfig(horizon_steps=4, grid_enabled=False))
    result = suite.run_all()
    assert len(result.rows) >= 20
    assert all("total_cost" in row for row in result.rows)
