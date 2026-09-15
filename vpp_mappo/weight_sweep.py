"""保留普通 MAPPO，多固定权重训练；只在验证集筛选策略档案。"""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
from .config import Config
from .runner import train, evaluate
from .data import CSVProfiles
from .objectives import OBJECTIVE_NAMES, CONTRACT_VERSION
from .pareto import non_dominated, hypervolume


def run(suite_path, output):
    suite=json.loads(Path(suite_path).read_text(encoding='utf-8'))
    cfg=Config.load(suite['config'])
    if not cfg.metrics_enabled or not cfg.safety or cfg.gamma != 1.0:
        raise ValueError('多目标配对实验要求 metrics_enabled、safety 且 gamma=1')
    if cfg.algorithm != 'mappo': raise ValueError('基础配置必须显式保留普通 mappo')
    preference=np.asarray(suite['weights'],dtype=float)
    if preference.ndim!=2 or preference.shape[1]!=3 or not np.isfinite(preference).all() or (preference<0).any() or not np.allclose(preference.sum(axis=1),1):
        raise ValueError('权重列表必须为归一化三维非负向量')
    reference=suite['hv_reference']; hypervolume([],reference)
    count=suite.get('evaluation_episodes',3)
    valid_seed=suite.get('validation_seed',200000); test_seed=suite.get('test_seed',300000)
    if type(count) is not int or count<1 or type(valid_seed) is not int or type(test_seed) is not int or min(valid_seed,test_seed)<0:
        raise ValueError('评估数量或种子不合法')
    if set(range(valid_seed,valid_seed+count)) & set(range(test_seed,test_seed+count)):
        raise ValueError('验证和测试场景种子必须独立')
    valid_csv=suite.get('validation_csv'); test_csv=suite.get('test_csv')
    if bool(valid_csv)!=bool(test_csv): raise ValueError('验证/测试必须同时使用 CSV 或同时使用合成数据')
    if bool(cfg.train_csv)!=bool(valid_csv): raise ValueError('本配对入口要求训练与评估的数据模式一致')
    if valid_csv:
        datasets=[CSVProfiles(path,cfg.horizon) for path in (cfg.train_csv,valid_csv,test_csv)]
        for i,a in enumerate(datasets):
            for b in datasets[i+1:]:
                if set(a.scenario_names)&set(b.scenario_names) or set(a.fingerprints)&set(b.fingerprints):
                    raise ValueError('训练、验证、测试之间存在重复日期或曲线')
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('输出目录非空')
    out.mkdir(parents=True,exist_ok=True)
    # 普通 MAPPO 永远是第一条独立实验，不能被用户偏好列表覆盖。
    experiments=[('mappo',replace(cfg,algorithm='mappo'))]
    experiments += [(f'weighted_{i:02d}',replace(cfg,algorithm='weighted_mappo',objective_weights=tuple(map(float,w)))) for i,w in enumerate(preference)]
    candidates=[]
    for label,config in experiments:
        train(config,out/label/'train')
        rows=evaluate(out/label/'train/latest.pt',out/label/'validation',count,valid_seed,config.device,valid_csv)
        vector=[float(np.mean([r[k] for r in rows])) for k in OBJECTIVE_NAMES]
        feasible=all(r['violations']==0 and r['ac_violations']==0 and r['ac_failed_steps']==0 and r['reserve_invalid_steps']==0 for r in rows)
        candidates.append(dict(label=label,algorithm=config.algorithm,weights=list(config.objective_weights),
                               validation_vector=vector,validation_feasible=feasible))
    front=non_dominated([r['validation_vector'] for r in candidates],[r['validation_feasible'] for r in candidates])
    archive=dict(contract=CONTRACT_VERSION,selected_labels=[candidates[i]['label'] for i in front],
        selection_split='validation only',hv_reference=reference,
        validation_hv=hypervolume([candidates[i]['validation_vector'] for i in front],reference),candidates=candidates,
        method='多个固定权重的独立策略；不是偏好条件 Pareto-MAPPO',
        budget=dict(per_policy_env_steps=cfg.episodes*cfg.horizon,total_env_steps=len(experiments)*cfg.episodes*cfg.horizon,
                    total_policies=len(experiments)),base_config=asdict(cfg),suite=suite)
    (out/'archive.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2),encoding='utf-8')
    test=[]
    for candidate in candidates:
        label=candidate['label']
        rows=evaluate(out/label/'train/latest.pt',out/label/'test',count,test_seed,cfg.device,test_csv)
        test.append(dict(label=label,selected_on_validation=label in archive['selected_labels'],
            vector=[float(np.mean([r[k] for r in rows])) for k in OBJECTIVE_NAMES],
            feasible=all(r['violations']==0 and r['ac_violations']==0 and r['ac_failed_steps']==0 and r['reserve_invalid_steps']==0 for r in rows)))
    # 测试不重新选策略；验证期已选但测试失效的候选不能计入可行 HV。
    frozen=[r['vector'] for r in test if r['selected_on_validation'] and r['feasible']]
    report=dict(policies=test,frozen_archive_test_hv=hypervolume(frozen,reference),hv_reference=reference,
                test_reselection=False,igd=None,igd_reason='未提供可信参考前沿，不计算 IGD')
    (out/'test_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return archive,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description='普通 MAPPO + 多固定权重独立策略对照')
    p.add_argument('--suite',default='configs/weight_sweep_smoke.json'); p.add_argument('--output',required=True)
    a=p.parse_args(); run(a.suite,a.output)
