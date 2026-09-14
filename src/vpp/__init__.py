"""按需导出模块，仿真核心无需安装 Web 服务或强化学习依赖。"""
from importlib import import_module

__version__ = '2.0.0'
__author__ = 'VPP Development Team'
__license__ = 'MIT'
__all__ = ['VirtualPowerPlant', 'VPPConfig', 'VPPError', 'optimization', 'models']
_EXPORTS = {'VirtualPowerPlant': ('.core', 'VirtualPowerPlant'), 'VPPConfig': ('.config', 'VPPConfig'), 'VPPError': ('.exceptions', 'VPPError'), 'optimization': ('.', 'optimization'), 'models': ('.', 'models')}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    obj = import_module(module, __name__)
    value = getattr(obj, symbol) if module != "." else import_module("." + symbol, __name__)
    globals()[name] = value
    return value
