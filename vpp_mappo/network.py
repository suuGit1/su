"""Baran–Wu 33 节点：统一无损线性模型与独立交流潮流审计。"""
import numpy as np
import pandapower as pp
from pandapower.networks import case33bw


class Network33:
    def __init__(self, spec):
        self.spec = spec
        self.net = case33bw()
        n = self.net
        # 原案例的热额定值不能作为实测导线容量；显式使用研究假设。
        n.line['max_i_ka'] = spec.assumed_line_ka
        self.lines = n.line[n.line.in_service].copy()
        if len(n.bus) != 33 or len(self.lines) != 32:
            raise ValueError('网络必须是 33 节点、32 条投入支路')
        self.base_p = np.zeros(33)
        self.base_q = np.zeros(33)
        for _, row in n.load.iterrows():
            self.base_p[int(row.bus)] += row.p_mw
            self.base_q[int(row.bus)] += row.q_mvar
        self.weights = self.base_p/self.base_p.sum()
        self.q_weights = self.base_q/self.base_p.sum()
        edges = [(int(r.from_bus), int(r.to_bus)) for _, r in self.lines.iterrows()]
        # 自动从平衡节点定向，不依赖支路表的排列顺序。
        adjacency = {i: [] for i in range(33)}
        for k, (a, b) in enumerate(edges):
            adjacency[a].append((b, k)); adjacency[b].append((a, k))
        self.paths = np.zeros((33, 32))
        seen = {0}; queue = [0]
        for a in queue:
            for b, k in adjacency[a]:
                if b in seen: continue
                seen.add(b); queue.append(b)
                self.paths[b] = self.paths[a]; self.paths[b, k] = 1
        if len(seen) != 33: raise ValueError('网络不连通')
        self.downstream = self.paths.T
        self.r = (self.lines.r_ohm_per_km*self.lines.length_km).to_numpy()
        self.x = (self.lines.x_ohm_per_km*self.lines.length_km).to_numpy()
        self.kv = float(n.bus.vn_kv.iloc[0])
        self.smax = np.sqrt(3)*self.kv*spec.assumed_line_ka
        # 动作是储能/EV 有功注入与按负荷比例分布的 DR，储能逆变器 Q=0。
        self.pa = np.zeros((33, 3)); self.qa = np.zeros((33, 3))
        self.pa[spec.ess_bus, 0] = -0.001; self.pa[spec.ev_bus, 1] = -0.001
        self.pa[:, 2] = -self.weights/1000; self.qa[:, 2] = -self.q_weights/1000
        self.pcoef = self.downstream@self.pa; self.qcoef = self.downstream@self.qa
        self.vcoef = -2*self.paths@(self.r[:, None]*self.pcoef+self.x[:, None]*self.qcoef)/self.kv**2
        self.generators = [pp.create_sgen(n, bus=b, p_mw=0, q_mvar=0) for b in range(1, 33)]

    def injections(self, row, action):
        p = self.weights*row['load_kw']/1000
        q = self.q_weights*row['load_kw']/1000
        for b in self.spec.pv_buses: p[b] -= row['pv_kw']/1000/len(self.spec.pv_buses)
        p[self.spec.wind_bus] -= row['wind_kw']/1000
        return p+self.pa@action, q+self.qa@action

    def affine(self, row):
        p, q = self.injections(row, np.zeros(3))
        bp, bq = self.downstream@p, self.downstream@q
        v = 1-2*self.paths@(self.r*bp+self.x*bq)/self.kv**2
        # 内接正方形是保守的视在功率约束：|P|,|Q| ≤ Smax/sqrt(2)。
        matrix = np.vstack((self.vcoef, self.pcoef, self.qcoef))
        offset = np.r_[v, bp, bq]
        lower = np.r_[np.full(33, self.spec.voltage_min**2), np.full(64, -self.smax/np.sqrt(2))]
        upper = np.r_[np.full(33, self.spec.voltage_max**2), np.full(64, self.smax/np.sqrt(2))]
        return matrix, lower-offset, upper-offset

    def linear(self, row, action):
        p, q = self.injections(row, action)
        bp, bq = self.downstream@p, self.downstream@q
        v2 = 1-2*self.paths@(self.r*bp+self.x*bq)/self.kv**2
        a, lo, hi = self.affine(row); value = a@action
        count = int(np.sum((value < lo-1e-7)|(value > hi+1e-7)))
        return dict(linear_min_vm_pu=float(np.sqrt(np.maximum(v2, 0)).min()),
                    linear_max_vm_pu=float(np.sqrt(np.maximum(v2, 0)).max()), linear_network_violations=count)

    def audit(self, row, action):
        n = self.net
        p, q = self.injections(row, action)
        # 将净注入直接映射为 sgen，等价于同母线的负荷与发电叠加。
        n.load[['p_mw','q_mvar']] = 0.0
        n.sgen.loc[self.generators, 'p_mw'] = -p[1:]
        n.sgen.loc[self.generators, 'q_mvar'] = -q[1:]
        try:
            pp.runpp(n, algorithm='bfsw', numba=False, calculate_voltage_angles=False, tolerance_mva=1e-8)
        except pp.LoadflowNotConverged:
            return dict(ac_converged=False, ac_violations=1, ac_min_vm_pu=None, ac_max_vm_pu=None,
                        ac_max_line_loading_pct=None, ac_loss_kw=None, ac_grid_kw=None)
        vm = n.res_bus.vm_pu.to_numpy()
        loading = n.res_line.loc[self.lines.index, 'loading_percent'].to_numpy()
        grid = float(n.res_ext_grid.p_mw.sum()*1000)
        count = int(np.sum(vm < self.spec.voltage_min-1e-7)+np.sum(vm > self.spec.voltage_max+1e-7)+np.sum(loading > 100+1e-6))
        count += int(grid > self.spec.grid_import+1e-5 or grid < -self.spec.grid_export-1e-5)
        return dict(ac_converged=True, ac_violations=count, ac_min_vm_pu=float(vm.min()), ac_max_vm_pu=float(vm.max()),
                    ac_max_line_loading_pct=float(loading.max()), ac_loss_kw=float(n.res_line.pl_mw.sum()*1000), ac_grid_kw=grid)
