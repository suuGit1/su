from vpp.dt_c3.rl import RLTrainConfig, DTC3MAPPOEMS, describe_torch_device, resolve_torch_device, torch


def test_device_resolver_accepts_auto_cpu_cuda():
    assert resolve_torch_device('auto') in {'cpu', 'cuda'} or resolve_torch_device('auto').startswith('cuda')
    assert resolve_torch_device('cpu') == 'cpu'
    resolved_cuda = resolve_torch_device('cuda')
    if torch is not None and torch.cuda.is_available():
        assert resolved_cuda.startswith('cuda')
    else:
        assert resolved_cuda == 'cpu'
    assert isinstance(describe_torch_device('auto'), str)


def test_rl_config_device_is_resolved():
    agent = DTC3MAPPOEMS(RLTrainConfig(episodes=1, hidden_dim=8, update_epochs=1, device='auto'))
    assert agent.config.device in {'cpu', 'cuda'} or agent.config.device.startswith('cuda')
