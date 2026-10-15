"""扣除与控制无关的固定时域外生基准；物理评价目标保持原样。"""
import numpy as np
from vpp_mappo.objectives import carbon_kg

def reference_vector(row,sell_ratio,dt_hours,scales):
    # 仅事后奖励核算使用实际外生数据，不加入actor或critic的观察。
    net=row['load_kw']-row['pv_kw']-row['wind_kw']
    cost=dt_hours*row['price']*(max(net,0)-sell_ratio*max(-net,0))
    carbon=carbon_kg(net,dt_hours,row['carbon_g_per_kwh'])
    return np.array([-cost,-carbon,0.])/np.asarray(scales)
