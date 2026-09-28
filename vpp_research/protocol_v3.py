"""冻结目标、信息条件、数据与实现版本，防止恢复时混合实验。"""
import hashlib
from pathlib import Path
from .dt import digest,VERSION as DT_VERSION
from .environment import OBS_VERSION,PCC_OBJECTIVE_VERSION,SCALES


def contract(config, plan):
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for package in ('vpp_research','vpp_mappo','vendor/mappo/onpolicy')
             for p in sorted((root/package).rglob('*.py'))}
    resources={}
    for key,value in config.__dict__.items():
        if isinstance(value,str) and value and Path(value).is_file():resources[key]=hashlib.sha256(Path(value).read_bytes()).hexdigest()
    record=dict(schema='experiment-contract-v3',objective_version=PCC_OBJECTIVE_VERSION,
        observation_version=OBS_VERSION,dt_version=DT_VERSION,source_digest=digest(sources),resource_hashes=resources,
        objectives=[dict(name='economic',direction='min',definition='AC并网成本+通信计算成本+终端惩罚'),
                    dict(name='carbon',direction='min',definition='并网进口碳排放+通信计算用电碳排放；不计出口减排信用'),
                    dict(name='reserve',direction='max',definition='同一时段起点的AC核验对称PCC备用能量；有限激活点检查')],
        vector_signs=[-1,-1,1],scales=SCALES.tolist(),hv_reference=plan['hv_reference'],
        information=dict(mpc='公开DT观察与因果预测；固定通信计算候选',milp_oracle='真实未来曲线与会话；非因果经济参考',
                         ordinary='公开DT观察；经济奖励',central='集中公开DT观察；经济奖励',
                         fixed='公开DT观察；固定加权奖励',ols='固定权重子策略；验证集选择',pareto='公开DT观察+偏好；向量critic/GAE'),
        shared_execution='统一设备约束、区间安全层、现场执行器、IEEE33 AC验收及PCC备用口径',
        budget_rule='学习方法共享名义训练回合上限；实际调用、成功、提交、重做、选择和测试单列；OLS无新角点允许早停',
        metric_rule='仅完整可行日期组进入前沿；IGD参考为验证经验前沿而非真实前沿；未认证与物理违反分别报告',
        real_input_fingerprint=plan['protocol']['fingerprint'],
        splits=dict(dt_days=plan['dt_days'],selection_dates=plan['selection_dates'],test_dates=plan['test_dates']),
        scope='规则式协调器；仿真C3；EV/DR范围由真实输入协议指定；训练与选择交互不宣称共同等预算')
    record['fingerprint']=digest(record)
    return record
