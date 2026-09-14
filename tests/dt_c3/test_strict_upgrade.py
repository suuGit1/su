from vpp.dt_c3 import DigitalTwinBenchmark, MILPEnergyManagement, MPCEnergyManagement, ADMMDERCoordinator
from vpp.dt_c3.communication import CommunicationConfig
from vpp.dt_c3.environment import PowerIoTVPPEnvironment, VPPEnvConfig


def test_traditional_baselines_smoke():
    for cls in [MILPEnergyManagement, MPCEnergyManagement, ADMMDERCoordinator]:
        env = PowerIoTVPPEnvironment(VPPEnvConfig(horizon_steps=3, communication=CommunicationConfig(seed=1), use_grid_constraints=False))
        metrics = cls().evaluate(env)
        assert "total_cost" in metrics
        assert metrics["avg_decision_time_s"] >= 0


def test_digital_twin_benchmark_outputs_rmse():
    true = [{"load_kw": 100.0 + i, "pv_kw": 20.0, "wind_kw": 10.0, "ess_soc": 0.5, "ev_soc": 0.6} for i in range(4)]
    obs = [true[0], None, true[2], true[3]]
    metrics = DigitalTwinBenchmark().run(true, obs)
    names = {m.name for m in metrics}
    assert "kalman" in names
    assert "proposed_dt_synchronizer" in names
    assert all(m.overall_rmse >= 0 for m in metrics)
