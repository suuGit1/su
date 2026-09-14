from vpp.dt_c3.communication import CommunicationChannel, CommunicationConfig
from vpp.dt_c3.ccc import C3ResourceManager
from vpp.dt_c3.digital_twin import DigitalTwinBenchmark
from vpp.dt_c3.optimization_baselines import MILPEnergyManagement, MPCEnergyManagement, ADMMDERCoordinator
from vpp.dt_c3.environment import PowerIoTVPPEnvironment, VPPEnvConfig


def test_c3_decision_changes_channel_behavior():
    ch = CommunicationChannel(CommunicationConfig(packet_loss_rate=0.2, bandwidth_kbps=256, upload_interval_steps=4, seed=1))
    report = C3ResourceManager().evaluate({"upload_priority": 1.0, "edge_cpu_fraction": 0.8, "offload_ratio": 0.2}, {"avg_aoi": 5, "packet_loss_ratio": 0.2}, {"safe": True, "risks": []})
    ch.apply_c3_report(report)
    st = ch.resource_state()
    assert st["runtime_bandwidth_kbps"] > 256
    assert st["runtime_upload_interval_steps"] <= 4
    assert st["runtime_packet_loss_multiplier"] < 1.0


def test_learning_digital_twin_estimators_present():
    true = [{"load_kw": 100 + i * 2, "pv_kw": 30 + i, "wind_kw": 15, "ess_soc": 0.5, "ev_soc": 0.6} for i in range(12)]
    obs = [row if i % 3 != 0 else None for i, row in enumerate(true)]
    metrics = DigitalTwinBenchmark(include_learning=True).run(true, obs)
    names = {m.name for m in metrics}
    assert "trained_gru_dt" in names
    assert "trained_lstm_dt" in names
    assert "trained_transformer_dt" in names


def test_traditional_optimization_final_baselines_run():
    for cls in [MILPEnergyManagement, MPCEnergyManagement, ADMMDERCoordinator]:
        env = PowerIoTVPPEnvironment(VPPEnvConfig(horizon_steps=4, use_grid_constraints=False))
        metrics = cls().evaluate(env)
        assert "total_cost" in metrics
        assert "solver_status" in metrics
