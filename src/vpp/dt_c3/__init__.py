"""DT-C3 extensions for Power IoT-enabled Virtual Power Plants.

This package adds the research modules required by the paper idea
"Digital Twin-Driven Communication-Computation-Control Co-Design for Safe
Energy Management in Power IoT-Enabled Virtual Power Plants".
"""

from .algorithms import DTC3SafeMARL, DTC3SafeMARLConfig
from .rl import RLTrainConfig, DTC3MAPPOEMS, MAPPOEMS, MADDPGEMS, PPOEMS, IACEMS, NoAoIMAPPOEMS
from .communication import CommunicationChannel, CommunicationConfig, CommunicationState
from .digital_twin import DigitalTwinSynchronizer, TwinConfig
from .environment import PowerIoTVPPEnvironment, VPPEnvConfig
from .experiments import DTC3ExperimentSuite, ExperimentSuiteConfig
from .advanced_experiments import AdvancedDTC3ExperimentSuite, AdvancedExperimentConfig
from .grid import IEEE33BusInterface, GridConfig
from .safety import SafetyLimits, SafetyShield
from .optimization_baselines import MILPEnergyManagement, MPCEnergyManagement, ADMMDERCoordinator
from .digital_twin import DigitalTwinBenchmark
from .ccc import C3ResourceManager, C3Config

__all__ = [
    "CommunicationChannel",
    "CommunicationConfig",
    "CommunicationState",
    "DigitalTwinSynchronizer",
    "TwinConfig",
    "SafetyShield",
    "SafetyLimits",
    "IEEE33BusInterface",
    "GridConfig",
    "PowerIoTVPPEnvironment",
    "VPPEnvConfig",
    "DTC3SafeMARL",
    "DTC3SafeMARLConfig",
    "RLTrainConfig",
    "DTC3MAPPOEMS",
    "MAPPOEMS",
    "MADDPGEMS",
    "PPOEMS",
    "IACEMS",
    "NoAoIMAPPOEMS",
    "MILPEnergyManagement",
    "MPCEnergyManagement",
    "ADMMDERCoordinator",
    "DigitalTwinBenchmark",
    "C3ResourceManager",
    "C3Config",
    "DTC3ExperimentSuite",
    "ExperimentSuiteConfig",
    "AdvancedDTC3ExperimentSuite",
    "AdvancedExperimentConfig",
]
