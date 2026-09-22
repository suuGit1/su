"""ACN 官方 JSON 导入：仅使用首次决策前可见申报，保留排除原因与参数假设。"""
import argparse
from collections import Counter
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
from vpp_mappo.flex_resources import validate_sessions


def utc(value):
    try:d=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:d=parsedate_to_datetime(value)
    if d.tzinfo is None:raise ValueError('EV 时间戳必须包含时区')
    return d.astimezone(timezone.utc)


def convert(raw,output,dt_hours=.25,max_kw=7.,source='https://ev.caltech.edu/dataset',replay_actual_departures=False):
    if not math.isfinite(dt_hours) or dt_hours<=0 or not math.isclose(24/dt_hours,round(24/dt_hours)):raise ValueError('步长必须整分一天')
    if not math.isfinite(max_kw) or max_kw<=0:raise ValueError('声明的桩容量必须正且有限')
    data=json.loads(Path(raw).read_text());items=data['_items'] if isinstance(data,dict) else data
    if not isinstance(items,list) or not items:raise ValueError('EV 数据为空或非完整会话列表')
    scenarios={};events={};excluded=Counter();kept=[];h=round(24/dt_hours);seconds=dt_hours*3600
    for item in items:
        try:
            arrival=utc(item['connectionTime']);day=arrival.replace(hour=0,minute=0,second=0,microsecond=0)
            a=math.ceil((arrival-day).total_seconds()/seconds);decision=day.timestamp()+a*seconds
            visible=[u for u in item.get('userInputs') or [] if utc(u['modifiedAt']).timestamp()<=decision]
            if not visible:excluded['no_visible_declared_demand']+=1;continue
            u=max(visible,key=lambda x:utc(x['modifiedAt']));departure=utc(u['requestedDeparture'])
            d=math.floor((departure-day).total_seconds()/seconds)
            if not 0<=a<d<=h:excluded['cross_day_or_short']+=1;continue
            energy=float(u['kWhRequested'])
            # 申报更新若已发生充电，不能使用整个会话的最终交付量推算剩余需求。
            if 'kWhDeliveredWhenModified' in u:energy-=float(u['kWhDeliveredWhenModified'])
            elif utc(u['modifiedAt'])>arrival:excluded['unknown_energy_before_update']+=1;continue
            sid=hashlib.sha256(str(item['sessionID']).encode()).hexdigest()[:20]
            record=dict(id=sid,arrival_step=a,departure_step=d,energy_kwh=max(0,energy),max_kw=max_kw)
            validate_sessions([record],h,dt_hours)
            event=None
            if replay_actual_departures:
                actual=utc(item['disconnectTime'])
                if actual<arrival:raise ValueError('实际离站早于接入')
                actual_step=max(a,math.floor((actual-day).total_seconds()/seconds))
                if actual_step<d:event=dict(id=sid,departure_step=actual_step)
            key=day.date().isoformat();scenarios.setdefault(key,[]).append(record)
            if event is not None:events.setdefault(key,[]).append(event)
            kept.append(dict(id=sid,declared_departure=departure.isoformat(),actual_departure=item.get('disconnectTime'),
                declaration_time=utc(u['modifiedAt']).isoformat()))
        except (ValueError,KeyError,TypeError) as exc:excluded[type(exc).__name__+':'+str(exc)[:100]]+=1
    if not scenarios:raise ValueError('没有可因果导入的会话；不能用最终交付量填补在线需求。排除统计：'+str(dict(excluded)))
    for records in scenarios.values():validate_sessions(records,h,dt_hours)
    result=dict(schema_version=1,energy_basis='grid_kwh',source=source,horizon=h,dt_hours=dt_hours,scenarios=scenarios,actual_departure_events=events)
    p=Path(output)
    if p.exists():raise ValueError('拒绝覆盖会话文件')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    audit=dict(source=source,raw_sha256=hashlib.sha256(Path(raw).read_bytes()).hexdigest(),input_count=len(items),accepted=len(kept),excluded=dict(excluded),
        assumptions=dict(max_kw=max_kw,departure='申报期限用于规划；真实提前离站事件到达后才更新现场状态' if replay_actual_departures else '仅申报期限回放；实际离站未启用',availability='到站向上取整；申报及实际离站向下取整，实际离站最早于首次控制步触发；亚步长时间为离散近似'),records=kept)
    p.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));return audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--max-kw',type=float,required=True);p.add_argument('--dt-hours',type=float,default=.25);p.add_argument('--replay-actual-departures',action='store_true');a=p.parse_args();print(convert(a.input,a.output,a.dt_hours,a.max_kw,replay_actual_departures=a.replay_actual_departures))
