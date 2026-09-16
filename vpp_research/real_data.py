"""真实 OPSD 负荷/风光/价格上的 DT 与控制验证；明确关闭缺数据的 EV。"""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from vpp_mappo.prepare_opsd import prepare
from vpp_mappo.data import CSVProfiles
from .validate_dt import collect,control
from .dt import fit,assess,save


def run(config,raw,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('真实数据输出目录非空')
    out.mkdir(parents=True,exist_ok=True)
    manifest=prepare(raw,out/'profiles',start='2019-01-01',train_end='2019-02-01',validation_end='2019-03-01',end='2019-04-01')
    c=replace(config,horizon=24,dt_hours=1.,reserve_hours=1.,synthetic_carbon_g_per_kwh=None)
    sources={k:out/'profiles'/(name+'.csv') for k,name in [('train','train'),('calibration','validation'),('test','test')]}
    scenarios=[s for path in sources.values() for s in CSVProfiles(path,24).scenario_names]
    c._ev_bundle=dict(schema_version=1,energy_basis='grid_kwh',source='显式无 EV 消融；尚无真实 EV 会话，不生成伪实测',horizon=24,dt_hours=1.,scenarios={s:[] for s in scenarios})
    counts={'train':24,'calibration':12,'test':8}
    datasets={mode:{k:collect(c,list(range(20000,20000+counts[k])),mode,path) for k,path in sources.items()} for mode in ('hold','physics')}
    results={}
    for method in ('hold','physics','residual'):
        d=datasets['hold' if method=='hold' else 'physics'];model=fit(d['train'],d['calibration'],method)
        model['source_manifest']=manifest;save(model,out/(method+'.json'))
        results[method]=dict(estimation=assess(model,d['test']),closed_loop=control(c,model,list(range(22000,22008)),csv_path=sources['test']))
    report=dict(source_manifest=manifest,scope='国家实测曲线缩放至 IEEE33；无 EV 消融；碳和硬件参数未实测；不是完整真实 VPP 三目标实验',results=results,
        missing_for_full_experiment=['接入时申报的真实 EV 需求与离站期限','同地区、同时段且口径一致的碳强度','通信/CPU/求解器设备测量'],
        coverage_note='跨月时序数据不保证交换性；区间覆盖仅作实测，不能自动继承名义覆盖保证')
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
