"""按需导出模块，仿真核心无需安装 Web 服务或强化学习依赖。"""
from importlib import import_module

__all__ = ['CommunicationChannel', 'CommunicationConfig', 'CommunicationState', 'DigitalTwinSynchronizer', 'TwinConfig', 'SafetyShield', 'SafetyLimits', 'IEEE33BusInterface', 'GridConfig', 'PowerIoTVPPEnvironment', 'VPPEnvConfig', 'DTC3SafeMARL', 'DTC3SafeMARLConfig', 'RLTrainConfig', 'DTC3MAPPOEMS', 'MAPPOEMS', 'MADDPGEMS', 'PPOEMS', 'IACEMS', 'NoAoIMAPPOEMS', 'MILPEnergyManagement', 'MPCEnergyManagement', 'ADMMDERCoordinator', 'DigitalTwinBenchmark', 'C3ResourceManager', 'C3Config', 'DTC3ExperimentSuite', 'ExperimentSuiteConfig', 'AdvancedDTC3ExperimentSuite', 'AdvancedExperimentConfig']
_EXPORTS = {'DTC3SafeMARL': ('.algorithms', 'DTC3SafeMARL'), 'DTC3SafeMARLConfig': ('.algorithms', 'DTC3SafeMARLConfig'), 'RLTrainConfig': ('.rl', 'RLTrainConfig'), 'DTC3MAPPOEMS': ('.rl', 'DTC3MAPPOEMS'), 'MAPPOEMS': ('.rl', 'MAPPOEMS'), 'MADDPGEMS': ('.rl', 'MADDPGEMS'), 'PPOEMS': ('.rl', 'PPOEMS'), 'IACEMS': ('.rl', 'IACEMS'), 'NoAoIMAPPOEMS': ('.rl', 'NoAoIMAPPOEMS'), 'CommunicationChannel': ('.communication', 'CommunicationChannel'), 'CommunicationConfig': ('.communication', 'CommunicationConfig'), 'CommunicationState': ('.communication', 'CommunicationState'), 'DigitalTwinSynchronizer': ('.digital_twin', 'DigitalTwinSynchronizer'), 'TwinConfig': ('.digital_twin', 'TwinConfig'), 'PowerIoTVPPEnvironment': ('.environment', 'PowerIoTVPPEnvironment'), 'VPPEnvConfig': ('.environment', 'VPPEnvConfig'), 'DTC3ExperimentSuite': ('.experiments', 'DTC3ExperimentSuite'), 'ExperimentSuiteConfig': ('.experiments', 'ExperimentSuiteConfig'), 'AdvancedDTC3ExperimentSuite': ('.advanced_experiments', 'AdvancedDTC3ExperimentSuite'), 'AdvancedExperimentConfig': ('.advanced_experiments', 'AdvancedExperimentConfig'), 'IEEE33BusInterface': ('.grid', 'IEEE33BusInterface'), 'GridConfig': ('.grid', 'GridConfig'), 'SafetyLimits': ('.safety', 'SafetyLimits'), 'SafetyShield': ('.safety', 'SafetyShield'), 'MILPEnergyManagement': ('.optimization_baselines', 'MILPEnergyManagement'), 'MPCEnergyManagement': ('.optimization_baselines', 'MPCEnergyManagement'), 'ADMMDERCoordinator': ('.optimization_baselines', 'ADMMDERCoordinator'), 'DigitalTwinBenchmark': ('.digital_twin', 'DigitalTwinBenchmark'), 'C3ResourceManager': ('.ccc', 'C3ResourceManager'), 'C3Config': ('.ccc', 'C3Config')}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    obj = import_module(module, __name__)
    value = getattr(obj, symbol) if module != "." else import_module("." + symbol, __name__)
    globals()[name] = value
    return value
