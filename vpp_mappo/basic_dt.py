"""基础物理 DT：只依赖已完成计算的遥测和候选指令历史，不访问物理环境。"""
import copy
import numpy as np


class BasicDT:
    def __init__(self,dispatch,flex,dt_hours,initial_payloads,mode='physics'):
        if mode not in ('physics','hold'):raise ValueError('DT 模式必须为 physics 或 hold')
        self.dispatch=dispatch;self.flex=flex;self.dt=dt_hours;self.mode=mode
        # 回合开始一次同步初始化，后续更新必须经过传输与计算。
        self.cache=copy.deepcopy(initial_payloads);self.stamps=np.zeros(6);self.history=[]

    def command(self,action):self.history.append(np.asarray(action,dtype=float).copy())

    def receive(self,packet):
        k=packet['channel']
        if packet['sampled']<=self.stamps[k]:return False
        self.cache[k]=copy.deepcopy(packet['payload']);self.stamps[k]=packet['sampled'];return True

    def estimate(self,step):
        data=copy.deepcopy(self.cache)
        if self.mode=='physics':
            for k in range(6):
                start=round(self.stamps[k]/(3600*self.dt))
                for t in range(start,step):
                    a=self.history[t]
                    if k==0:
                        soc=self.dispatch.next_soc([data[0]['soc'],0],[a[0],0],self.dt)[0]
                        data[0]['soc']=float(np.clip(soc,self.dispatch.soc_min[0],self.dispatch.soc_max[0]))
                    elif k==1:
                        left=max(0,a[1])*self.dt
                        for s in sorted(data[1]['sessions'],key=lambda s:(s['departure_step'],s['id'])):
                            if not s['arrival_step']<=t<s['departure_step']:continue
                            delivered=min(left,s['remaining_kwh'],s['max_kw']*self.dt)
                            s['remaining_kwh']-=delivered;left-=delivered
                    elif k==2:
                        data[2]['backlog']=float(np.clip(data[2]['backlog']+a[2]*self.dt,0,self.flex.dr_backlog_kwh))
                        data[2]['shifted']+=max(0,a[2])*self.dt
                    elif k==3:data[3]['shed_used']+=max(0,a[3])*self.dt
        # 到期从已知会话集合移除；绝不推断未上传的新接入车辆。
        data[1]['sessions']=[s for s in data[1]['sessions'] if s['arrival_step']<=step<s['departure_step']]
        return data

    def age(self,now):return np.maximum(0,now-self.stamps)
