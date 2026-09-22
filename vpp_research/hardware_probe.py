"""记录当前运行主机可验证的计时/能量接口；缺失测量不以假设补齐。"""
import argparse
import json
import platform
from pathlib import Path
import time


def probe():
    paths=sorted(Path('/sys/class/powercap').glob('**/energy_uj'))
    counters=[]
    for p in paths:
        try:
            counters.append(dict(path=str(p),energy_uj=int(p.read_text()),
                                 max_energy_range_uj=int((p.parent/'max_energy_range_uj').read_text())))
        except (OSError,ValueError):continue
    clock=time.get_clock_info('perf_counter')
    return dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
        clock=dict(name='perf_counter',resolution_seconds=clock.resolution,monotonic=clock.monotonic),
        energy_measurement_available=bool(counters),energy_counters=counters,
        provenance='当前执行主机；不是目标边缘设备，也没有测量物理通信链路',
        limitation='若有计数器，系统级能量仍需扣除空闲基线及并发任务；无计数器则能耗不可实测，禁止以CPU时间替代能量')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(probe(),ensure_ascii=False,indent=2))
