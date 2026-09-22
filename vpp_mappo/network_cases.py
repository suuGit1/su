"""从有来源记录的原始物理单位表构建 69 节点算例。"""
import json
from pathlib import Path
import pandapower as pp


def case69():
    data = json.loads((Path(__file__).parent/'data/case69.json').read_text())
    net = pp.create_empty_network(sn_mva=data['base_mva'])
    for bus, p, q in data['bus']:
        pp.create_bus(net, vn_kv=data['base_kv'], index=int(bus)-1)
        if p or q:
            pp.create_load(net, int(bus)-1, p_mw=p/1000, q_mvar=q/1000)
    pp.create_ext_grid(net, 0, vm_pu=1.)
    for a, b, r, x in data['branch']:
        # 长度取 1 km，仅为了将总欧姆数传给线路 API，并非真实线路长度。
        pp.create_line_from_parameters(net, int(a)-1, int(b)-1, length_km=1.,
            r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0., max_i_ka=.4)
    return net
