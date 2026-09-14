from vpp.dt_c3.communication import CommunicationChannel, CommunicationConfig


def test_delay_and_aoi_are_reported():
    ch = CommunicationChannel(CommunicationConfig(max_delay_steps=2, fixed_delay_steps=2, seed=1))
    r0 = ch.transmit("ess", {"soc": 0.5}, 0)
    r1 = ch.transmit("ess", {"soc": 0.6}, 1)
    r2 = ch.transmit("ess", {"soc": 0.7}, 2)
    assert r0.communication.aoi >= 1
    assert r1.communication.aoi >= 1
    assert r2.observation is not None
    assert r2.communication.aoi == 2


def test_packet_loss_uses_missing_flag_without_previous_delivery():
    ch = CommunicationChannel(CommunicationConfig(packet_loss_rate=1.0, seed=1))
    result = ch.transmit("pv", {"power_kw": 10.0}, 0)
    assert result.observation is None
    assert result.communication.missing is True
