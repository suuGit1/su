from vpp.dt_c3.communication import CommunicationConfig
from vpp.dt_c3.environment import PowerIoTVPPEnvironment, VPPEnvConfig
from vpp.dt_c3.rl import DTC3MAPPOEMS, PPOEMS, IACEMS, MADDPGEMS, RLTrainConfig, torch_available


def make_env(seed=1):
    return PowerIoTVPPEnvironment(
        VPPEnvConfig(
            horizon_steps=6,
            seed=seed,
            communication=CommunicationConfig(max_delay_steps=1, fixed_delay_steps=1, packet_loss_rate=0.05, seed=seed),
            use_grid_constraints=False,
        )
    )


def test_neural_actor_critic_smoke_training():
    if not torch_available():
        return
    for cls in [DTC3MAPPOEMS, PPOEMS, IACEMS, MADDPGEMS]:
        model = cls(RLTrainConfig(episodes=1, seed=1, hidden_dim=16, update_epochs=1, batch_size=16))
        trace = model.train(lambda: make_env(1), episodes=1)
        metrics = model.evaluate(make_env(2))
        assert len(trace.episode_rewards) == 1
        assert "total_cost" in metrics
        assert metrics["avg_decision_time_s"] >= 0
