"""统一真实数据目录与资源参数接入；缺失日曲线不得退回合成数据。"""
from dataclasses import asdict,replace
import json
from pathlib import Path
from vpp_mappo.config import Config
from vpp_mappo.data import CSVProfiles
from vpp_mappo.flex_resources import FlexSpec
from .dr_import import conservative_flex
from .dt import digest


def connect(folder,config=None,ev_sessions=None,dr_mode='off'):
    root=Path(folder).resolve()
    catalog=json.loads((root/'catalog.json').read_text())
    paths={k:root/catalog['default_profiles']/(k+'.csv') for k in ('train','validation','test')}
    manifest=json.loads((paths['train'].parent/'manifest.json').read_text())
    datasets={k:CSVProfiles(path,manifest['horizon']) for k,path in paths.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if set(datasets[a].scenario_names)&set(datasets[b].scenario_names) or set(datasets[a].fingerprints)&set(datasets[b].fingerprints):
            raise ValueError('训练/校准/测试存在日期或日曲线重叠')
    if any('carbon_g_per_kwh' not in p or 'carbon_observed_g_per_kwh' not in p for d in datasets.values() for p in d.profiles):
        raise ValueError('真实三目标必须包含核算碳与因果观察碳列')
    base=config or Config.load('configs/integrated_ieee33.json')
    c=replace(base,horizon=manifest['horizon'],dt_hours=manifest['dt_hours'],reserve_hours=manifest['dt_hours'],
              train_csv=str(paths['train']),synthetic_carbon_g_per_kwh=None)
    names=[n for d in datasets.values() for n in d.scenario_names]
    if ev_sessions:
        bundle=json.loads(Path(ev_sessions).read_text())
        if bundle.get('horizon')!=c.horizon or bundle.get('dt_hours')!=c.dt_hours:raise ValueError('EV 时间粒度不匹配')
        if any(n not in bundle.get('scenarios',{}) for n in names):raise ValueError('EV 文件缺少能源数据中的日期')
        ev_scope='用户提供会话，地域/需求口径以其source为准'
    else:
        bundle=dict(schema_version=1,energy_basis='grid_kwh',source='显式关闭EV：未取得真实完整会话',
                    horizon=c.horizon,dt_hours=c.dt_hours,scenarios={n:[] for n in names})
        ev_scope='disabled_missing_real_sessions'
    c._ev_bundle=bundle
    base_flex=asdict(FlexSpec.load(c.flex_spec));dr={}
    if dr_mode=='off':
        for key in ('dr_shift_kw','dr_repay_kw','dr_backlog_kwh','dr_shift_budget_kwh','dr_shed_kw','dr_shed_budget_kwh'):base_flex[key]=0.
        dr=dict(mode='off',reason='没有同地区真实可控DR参数，不自动代入合成DR')
    elif dr_mode=='sce-derated':
        events=json.loads((root/'dr/sce/events.json').read_text())
        base_flex,dr=conservative_flex(base_flex,events)
        for key in ('dr_shift_kw','dr_repay_kw','dr_backlog_kwh','dr_shift_budget_kwh'):base_flex[key]=0.
        dr.update(mode=dr_mode,scope='明确跨地区敏感性：GB曲线+SCE历史训练响应分位数；不模拟SCE实站')
    else:raise ValueError('未知 DR 模式')
    c._flex_record=base_flex
    protocol=dict(schema='real-inputs-contract-v1',profiles={k:dict(sha256=d.sha256,days=d.scenario_names) for k,d in datasets.items()},
        horizon=c.horizon,dt_hours=c.dt_hours,ev_scope=ev_scope,ev_digest=digest(bundle),dr=dr,flex=base_flex,
        limits=['真实国家能源/碳曲线驱动仿真，非IEEE33实测','储能/线路/C3能耗与设备能力仍为研究假设'])
    protocol['fingerprint']=digest(protocol)
    return c,paths,datasets,protocol
