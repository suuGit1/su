# 第四阶段：三目标口径与普通 MAPPO 对照保留

本阶段把经济、碳排、灵活性写成可执行指标，新增固定权重多目标训练与非支配策略档案。普通 MAPPO 的算法名、训练入口、检查点与单目标奖励保留。未来偏好条件 Pareto-MAPPO 必须作为新增算法实现，不允许覆盖或把 weighted_mappo 改名冒充。

## 实际借鉴的内容

| 来源 | 已核对实现 | 本项目处理 |
|---|---|---|
| [CrazyRL MOMAPPO](https://github.com/ffelten/CrazyRL/blob/7a991d0a15a90c6a7a81491d76afd4e7a5d080f8/learning/fulljax/momappo_fulljax.py) | 对向量奖励作固定线性加权，并用 JAX vmap 并行训练多组权重 | 独立实现 PyTorch 多权重实验编排，继续使用官方 MAPPO 更新器 |
| [MOMAland MOMAPPO](https://github.com/Farama-Foundation/MOMAland/blob/54b61a1f9d3ecdec07cc4ebba909a3c718819f75/momaland/learning/cooperative_momappo/continuous_momappo.py) | 支持 OLS 或均匀权重，逐权重训练标量化 MAPPO | OLS 是后续可增加的权重选择对照，本阶段仅显式给定权重 |
| [MOMAland 环境接口](https://github.com/Farama-Foundation/MOMAland/blob/54b61a1f9d3ecdec07cc4ebba909a3c718819f75/momaland/utils/env.py) | 每智能体有 reward_space，奖励为向量 | 新增独立 VPPMOParallelEnv，供外部多目标算法适配 |

两者都是有价值的参考，但不能把多组固定权重策略称为已经实现单网络偏好条件策略，线性权重也不保证覆盖非凸前沿。本次没有复制上述仓库源码，不增加 JAX 或完整 MOMAland 安装依赖；参考提交已固定。普通 MAPPO 仍来自原先锁定的 marlbenchmark/on-policy。

## 运行

完整分支为 research/stage4-objective-baselines。按根 README 激活 Python 环境后：

```bash
python -m pip install -r requirements-grid.txt
python -m vpp_mappo train --config configs/objectives_smoke.json --algorithm mappo --output output/objective_mappo
python -m vpp_mappo.weight_sweep --suite configs/weight_sweep_smoke.json --output output/weight_sweep
```

weight_sweep 固定包含一条普通 mappo，外加配置中每组权重对应的一条独立 weighted_mappo。默认示例是普通 MAPPO 加四组权重，每条训练两个四步回合。每条策略使用相同训练场景、验证场景、测试场景和物理参数；各自产生 train、validation、test 子目录。archive.json 保存验证集筛选结果；test_report.json 评价已冻结的档案，不在测试集重新选策略。即使普通 MAPPO 被其他策略支配，其测试结果仍独立保留。

每策略预算相同不等于整个多策略方法总预算相同，archive.json 明确记录每策略与总环境步数。正式比较单策略与策略集合时，必须另做总预算匹配实验及多个训练种子。当前示例不是性能实验。

已有第二、三阶段不启用 metrics_enabled 的配置继续运行。三目标模式增加当前碳因子这一公共观测，因此输入从 12 维变为 13 维；公平对照中的普通 MAPPO 和 weighted_mappo 都使用同样的 13 维输入。不能把旧 12 维权重直接用于新输入。

## 三目标可执行定义

定义版本：economic-carbon-reserve-v1。所有向量均采用最大化约定，次序固定为经济、低碳、灵活性。

1. 经济：负的当步经济费用，真实日末另加 SOC 短缺罚项。除以 objective_scales[0]。
2. 碳排：max(主网进口 kW,0) × dt_hours × 碳强度 gCO2/kWh / 1000，单位 kgCO2；目标取负后除以 objective_scales[1]。
3. 灵活性：执行后状态能提供的对称备用 kW × dt_hours，除以 objective_scales[2]。

碳边界仅为进口电力运行排放：出口不抵扣，储能放电不重复记排放，未计电池制造和退化的生命周期排放。交流含损耗的 ac_carbon_kg 单独输出；优化目标仍采用无损线性边界。不能据此宣称边际减排或生命周期净零。

灵活性目前是明确限定的研究指标：保持当前风光/负荷不变，以执行后 SOC 为初值，在 reserve_hours 的单段持续时间内，通过共享 SOC、功率和 LinDistFlow 约束分别求最小/最大主网功率。相对于刚执行的基准功率，取可上调和下调余量的较小值。若基准功率无法由执行后状态持续维持，则对称备用为零。若本步已违反物理约束，则 reserve_valid=false，指标占位为零，档案筛选拒绝该候选。

该指标不是实际激活电量、不是多时段备用承诺，也不是 AC 已认证备用；还未纳入 EV 离站、DR 反弹及外部轨迹不确定性。随着这些物理约束补齐，指标也必须升级版本，并对所有算法重新计算。

备用标签每步额外调用两次 MILP，时间单列 reserve_solver_seconds；这是指标计算成本，不能把含此开销的训练时间直接当作策略在线推理时间。后续 C3 需要拆分真实控制、安全校验和离线评价计时。

普通 mappo 的 reward 仍为原先经济与违规罚项，不受 objective_weights 改变。weighted_mappo 使用三维归一化向量与固定权重的内积，违规罚项独立扣除，不能被零经济权重消掉。采用本示例尺度、权重 [1,0,0] 时，weighted_mappo 与普通 mappo 的奖励相等，并有回归测试验证。

## 数据来源纪律

示例配置显式使用 synthetic_carbon_g_per_kwh=300，是合成假设，输出标注 explicit_synthetic_assumption，不代表任何地区实测碳强度。真实 CSV 必须提供 carbon_g_per_kwh 列；缺失时拒绝启动三目标实验，不自动用示例数值补齐。

OPSD 第三阶段预处理文件本身没有碳列，仍可用于原普通 MAPPO。真实三目标实验必须先接入同地区、同时间、可追溯的碳数据并更新 manifest，不能直接修改 CSV 后忽略哈希不匹配。需要保留原始来源、时区、发布时间和预测/事后估计区分。本阶段仅完成碳列接口，未完成真实碳数据下载和地区匹配。

训练、验证、测试仍分别检查日期和逐日基础曲线重叠。策略档案只在验证集上筛选；所有实际违规、AC 不收敛或备用指标无效的候选均不能进入可行档案。均值非支配不等于逐日不被支配，也不证明随机策略在分布外安全。

## Pareto 评价

pareto.py 提供去重的非支配筛选、三维精确盒并集 HV，以及显式参考集 IGD。候选必须先通过可行性筛选。尺度和 HV 参考点在实验前统一固定；本示例参考点只用于运行检查，不应直接当作正式论文参数。仅支配参考点的正体积盒计入 HV。

没有参考前沿时不计算 IGD，报告 null 和原因。多权重解集不是已知真实前沿。测试评价使用验证期选定策略，未对测试结果再次执行策略选择；测试失效的候选不贡献可行 HV，同时仍保留报告。

## MOMAland 风格并行接口

此接口独立于现有官方 MAPPO 主循环，需要可选依赖：

```bash
python -m pip install -r requirements-mo-api.txt
python validation/test_stage4.py
```

```python
from vpp_mappo.config import Config
from vpp_mappo.mo_parallel import VPPMOParallelEnv

env = VPPMOParallelEnv(Config.load('configs/objectives_smoke.json'))
observations, infos = env.reset(seed=1)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    observations, rewards, terminated, truncated, infos = env.step(actions)
env.close()
```

每智能体 reward_space 为三维，团队目标均分给 ESS/EV/DR，按智能体求和可恢复团队向量，防止重复计三次。共享中央状态通过 state() 返回。安全违规罚项保存在 info 中，外部训练器若需要，应独立处理；不能假设原生向量奖励已经混入罚项。接口基于 PettingZoo ParallelEnv，借鉴 MOMAland 约定；没有声称所有 MOMAland 学习脚本可以零修改直接使用。

## 后续实施顺序

本阶段完成指标口径与多权重参考，并没有完成整个研究计划。下一优先事项仍是 EV 到离站/需求、弃电与 DR 能量约束，并同步更新 MILP/MPC。之后实现通信与计算事件、Hybrid DT，再实现单网络偏好条件 Pareto-MAPPO、增强安全层和完整多种子实验。可执行普通 MAPPO 与固定权重对照必须持续保留。完整验收清单见 research_plan.json。
