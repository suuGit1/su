"""第三阶段统一物理环境；MAPPO 与优化对照使用同一执行器。"""
import numpy as np
from gymnasium.spaces import Box
from .data import CSVProfiles
from .dispatch import DispatchSpec
from .network import Network33
from .optimization import solve_dispatch


class GridVPPAdapter:
    num_agents = 3
    observation_space = Box(-np.inf, np.inf, (12,), dtype=np.float32)
    share_observation_space = Box(-np.inf, np.inf, (36,), dtype=np.float32)
    action_space = Box(-np.inf, np.inf, (1,), dtype=np.float32)

    def __init__(self, config, csv_path=None):
        self.config = config
        self.spec = DispatchSpec.load(getattr(config, '_dispatch_record', None) or config.dispatch_spec)
        if config.network_model != self.spec.network_name:
            raise ValueError("配置网络与资源参数中的网络算例不一致")
        self.network = Network33(self.spec)
        self.data = CSVProfiles(csv_path, config.horizon) if csv_path else None
        if config.metrics_enabled:
            self.observation_space = Box(-np.inf, np.inf, (13,), dtype=np.float32)
            self.share_observation_space = Box(-np.inf, np.inf, (39,), dtype=np.float32)
            if self.data and any('carbon_g_per_kwh' not in p for p in self.data.profiles):
                raise ValueError('真实 CSV 缺少 carbon_g_per_kwh；不得用示例碳因子冒充实测/来源估计值')
            if not self.data and config.synthetic_carbon_g_per_kwh is None:
                raise ValueError('合成三目标示例必须显式配置 synthetic_carbon_g_per_kwh')
        if self.data and self.data.provenance and self.data.provenance['dt_hours'] != config.dt_hours:
            raise ValueError('真实 CSV 的时间步长与调度配置不一致')
        if config.delay_steps or config.packet_loss or config.digital_twin:
            raise ValueError('IEEE33 统一对照暂限无延迟、无丢包、无 DT；避免优化器额外读取真实状态')

    def reset(self, scenario_seed, profile_index=0):
        self.t = 0; self.soc = np.array(self.spec.initial_soc, dtype=float)
        if self.data: self.profiles = self.data.get(profile_index)
        else:
            rng = np.random.default_rng(scenario_seed)
            hour = np.arange(self.config.horizon)*self.config.dt_hours
            self.profiles = dict(load_kw=1700+200*np.sin(hour*2*np.pi/24)+rng.uniform(-50,50,len(hour)),
                pv_kw=700*np.maximum(0,np.sin((hour-6)*np.pi/12)), wind_kw=rng.uniform(100,250,len(hour)),
                price=np.where((hour>=17)&(hour<21),0.3,0.1))
            if self.config.metrics_enabled:
                self.profiles['carbon_g_per_kwh'] = np.full(self.config.horizon, self.config.synthetic_carbon_g_per_kwh)
        return self.encode()

    def row(self, t=None):
        t = self.t if t is None else t
        return {k:float(v[t]) for k,v in self.profiles.items()}

    def encode(self):
        if self.t == self.config.horizon:
            return np.zeros((3,*self.observation_space.shape),np.float32),np.zeros((3,*self.share_observation_space.shape),np.float32)
        row = self.row(); linear = self.network.linear(row,np.zeros(3))
        common = [self.t/self.config.horizon,row['price']/0.3,(row['load_kw']-row['pv_kw']-row['wind_kw'])/5000,
                  row['pv_kw']/1000,row['wind_kw']/1000,linear['linear_min_vm_pu'],linear['linear_max_vm_pu']]
        if self.config.metrics_enabled:
            common.append(row['carbon_g_per_kwh']/1000)
        own = [(self.soc[0],self.spec.target_soc[0]),(self.soc[1],self.spec.target_soc[1]),(row['load_kw']/5000,0)]
        obs = np.array([common+list(own[i])+np.eye(3)[i].tolist() for i in range(3)],np.float32)
        return obs,np.repeat(obs.reshape(1,-1),3,axis=0)

    def physical_violations(self, row, action):
        s = self.spec; next_soc = s.next_soc(self.soc,action,self.config.dt_hours)
        grid = row['load_kw']-row['pv_kw']-row['wind_kw']-sum(action)
        count = int(np.sum(np.abs(action[:2])>np.array(s.power_max)+1e-5))
        count += int(action[2]<-1e-5 or action[2]>min(s.dr_max,row['load_kw'])+1e-5)
        count += int(np.sum(next_soc<np.array(s.soc_min)-1e-7)+np.sum(next_soc>np.array(s.soc_max)+1e-7))
        count += int(grid>s.grid_import+1e-5 or grid < -s.grid_export-1e-5)
        return count+self.network.linear(row,action)['linear_network_violations']

    def step(self, actions):
        raw = np.asarray(actions)
        if raw.shape != (3,1) or not np.isfinite(raw).all(): raise ValueError('动作必须为有限 (3,1) 数组')
        x = np.tanh(raw[:,0])
        physical = np.r_[x[:2]*self.spec.power_max,(x[2]+1)*self.spec.dr_max/2]
        return self.step_physical(physical)

    def step_physical(self, action):
        if self.t >= self.config.horizon: raise RuntimeError('回合已结束，请 reset')
        action = np.asarray(action,dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all(): raise ValueError('物理动作必须为有限三维数组')
        row = self.row(); pre = self.physical_violations(row,action); proposal = action.copy()
        if self.config.safety and pre:
            plan,_ = solve_dispatch(self.spec,self.network,[row],self.soc,self.config.dt_hours,0,self.config.solver_time_limit,proposal=action)
            action = plan[0]
        count = self.physical_violations(row,action)
        grid = row['load_kw']-row['pv_kw']-row['wind_kw']-sum(action)
        self.soc = self.spec.next_soc(self.soc,action,self.config.dt_hours)
        cost = self.spec.cost(grid,action,row['price'],self.config.dt_hours)
        self.t += 1; done = self.t == self.config.horizon
        terminal = self.config.terminal_soc_penalty*np.maximum(np.array(self.spec.target_soc)-self.soc,0).sum() if done else 0.0
        audit = self.network.audit(row,action)
        reward = -(cost+terminal+self.config.violation_penalty*count)/self.config.reward_scale
        info = dict(cost=cost,reward=float(reward),constraint_violations=count,pre_shield_violations=pre,
            shield_l1_kw=float(np.abs(action-proposal).sum()),ess_soc=float(self.soc[0]),ev_soc=float(self.soc[1]),
            ess_power_kw=float(action[0]),ev_power_kw=float(action[1]),dr_kw=float(action[2]),grid_power_kw=float(grid),
            terminal_penalty=float(terminal),**self.network.linear(row,action),**audit)
        info['ac_cost'] = self.spec.cost(audit['ac_grid_kw'],action,row['price'],self.config.dt_hours) if audit['ac_converged'] else None
        if self.config.metrics_enabled:
            from .objectives import step_metrics, OBJECTIVE_NAMES
            info.update(step_metrics(self.config,self.spec,self.network,row,self.soc,info))
            if self.config.algorithm == 'weighted_mappo':
                reward = float(np.dot(self.config.objective_weights,[info[k] for k in OBJECTIVE_NAMES]))
                reward -= self.config.violation_penalty*count/self.config.reward_scale
                info['reward'] = reward
        obs,share = self.encode()
        return obs,share,np.full((3,1),reward,np.float32),done,info
