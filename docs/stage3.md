# 第三阶段：IEEE33、统一 MILP/MPC 与真实曲线

## 选择的来源

基础网络直接调用 pandapower.networks.case33bw()。严格名称为 Baran–Wu 33 节点配电网，通常称 IEEE33；来源是 Baran & Wu (1989)，经 MATPOWER 和 pandapower 提供。包含 33 母线、32 条投入径向支路，原有 5 条联络线保持断开，不改变原始线路阻抗。

- [pandapower 案例说明](https://pandapower.readthedocs.io/en/v2.14.11/networks/power_system_test_cases.html)
- [MATPOWER case33bw](https://github.com/MATPOWER/matpower/blob/master/data/case33bw.m)
- [用户提供的网络脚本](https://github.com/xuwkk/gen_pandapower_pv/blob/master/bus33bw.py)

用户提供的脚本确实使用 case33bw，并在索引 12、17、21、24、28、32 加入光伏。本版参考这些布点，通过 pandapower 公开 API 独立构建，不读取旧 pickle，也没有复制该仓库源码。这不是对该论文 Volt/Var 无功控制算法的复现。本版仍聚焦有功能量调度。

## 安装和快速运行

从完整新分支获取代码：

```bash
git clone --branch research/stage3-ieee33-mpc https://github.com/suuGit1/su.git
cd su
```

已有本仓库可 fetch 后切换同名分支。按根 README 创建并激活环境，安装 CPU PyTorch 后：

```bash
python -m pip install -r requirements-grid.txt
python validation/test_stage3.py
python -m vpp_mappo train --config configs/ieee33_smoke.json --output output/grid_smoke
python -m vpp_mappo.compare --checkpoint output/grid_smoke/latest.pt --episodes 2 --output output/grid_compare
```

新配置显式设置 network_model=ieee33。第二阶段 aggregate 配置继续可用，但不能与第三阶段不同容量/网限的结果混合比较。第三阶段配置和资源快照保存到检查点，评估无需再次读取原始资源配置文件。新 12 维观测对应新的网络参数，不能直接把第二阶段 10 维权重用于 IEEE33。

## 真实数据端到端命令

来源：[OPSD Time Series 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06/)。[小时 CSV 直接下载](https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv)，130,339,665 字节，约 124 MiB。也可让程序自动下载并检查固定 SHA-256：

```bash
python -m vpp_mappo.download_opsd
python -m vpp_mappo.prepare_opsd --input data/raw/time_series_60min_singleindex.csv --output data/opsd2019
python -m vpp_mappo train --config configs/ieee33_hourly.json --csv data/opsd2019/train.csv --episodes 2 --output output/real_smoke
python -m vpp_mappo.compare --checkpoint output/real_smoke/latest.pt --csv data/opsd2019/test.csv --episodes 2 --output output/real_compare
```

下载失败时可在浏览器手动下载到同一路径，再执行预处理。大数据文件与权重不写入 Git；下载器、原始哈希、转换代码和验证记录可复现数据。两个训练回合仅用于检查运行，不用于论文比较。正式训练使用新输出目录和更大 episodes，多个训练种子；先用 validation.csv 选参数，最后固定后才使用 test.csv。compare 可增加 --episodes 61 覆盖全部保留测试日。

本次已真实下载并转换 2019 年数据：

| 分组 | UTC 日期范围 | 完整天数 |
|---|---|---:|
| 训练 | 2019-01-01 至 2019-08-31 | 242 |
| 验证 | 2019-09-01 至 2019-10-31 | 61 |
| 测试 | 2019-11-01 至 2019-12-31 | 61 |

默认是 24×1 小时，不能把小时数据直接当成 96×15 分钟。本次剔除 2019-03-31；不补零、不使用测试期插值。预处理检查完整 UTC 时间戳、缺失和负功率。训练集拟合负荷/PV/风电峰值映射至 1800/700/300 kW，验证测试使用同一系数且不裁剪超峰值。电价 EUR/MWh 转 EUR/kWh；负电价保留。数据时间间隔、来源、缩放、原始与分组哈希记录在 manifest.json，运行时验证匹配。训练测试日期及逐日曲线指纹会检查重叠。

这是国家级真实曲线缩放驱动的研究馈线仿真，不是 IEEE33 现场量测。负荷和风光取德国总量，价格取 DE_LU 分区；空间负荷采用案例比例，PV 平均分布，风电放在索引 32。跨地理层级映射及装机容量均是研究假设。正式论文需引用 OPSD 数据包及其原始 ENTSO-E 来源，遵守数据源条款；不将数据冒称本项目自产。

## 统一约束和经济目标

所有第三阶段控制器使用 DispatchSpec、Network33、GridVPPAdapter。参数位于 configs/dispatch_ieee33.json：容量 ESS 500 / EV 1200 kWh，功率 250 / 300 kW，充放效率均 0.95，初始和目标 SOC 均 0.55；SOC 边界分别 [0.1,0.9]、[0.2,0.9]。进口/出口限额为 5000/3000 kW，DR 上限 150 kW 且不超过当步负荷。更换参数需对所有算法一致。

正储能动作表示放电；SOC 更新为 SOC_next = SOC + (eta*charge - discharge/eta)*dt/capacity。MILP 使用二进制模式阻止同时充放电、同时购售电，在负电价下仍成立。购电减售电等于负荷减风光、DR 和储能净注入。DR 按母线基准负荷比例分摊，同时按原功率因数削减 Q；逆变器不调无功。

经济费用为电能交易 + 0.01 EUR/kWh 储能吞吐成本 + 0.03 EUR/kWh DR 成本；售电价格为购电价的 0.65。日末不足目标 SOC 以配置罚系数软惩罚。系数是显式研究假设，不是实测市场服务成本。记录 cost 与 cost+terminal_penalty 的 objective，reward 还含违规罚项并缩放。日末目标不是离站硬约束。

网络采用径向无损 LinDistFlow：支路 P/Q 为下游净需求总和，平方电压沿路径按 2(rP+xQ)/Vbase² 下降。电压范围 0.95–1.05 pu。支路采用 |P|、|Q| ≤ Smax/sqrt(2) 的保守线性盒约束。案例没有可直接用于实测热容量的额定值，因此统一显式假设线路 0.4 kA；Smax 用额定电压换算。此假设必须在论文中说明并做灵敏度分析，不是 IEEE33 标准热额定值。

MAPPO 原始高斯动作先映射到物理功率；若违反共享约束，单步 MILP 做归一化 L1 最近可行动作投影。优化对照与学习策略经过同一个物理执行器；可行计划不会被再次修改。每次 MILP 记录状态、最优标记、可行解目标、下界、gap、运行时间及最大残差。没有可行解时明确失败，不用零动作冒充成功。

## 两种优化对照的信息条件

| 对照 | 预测输入 | 执行方式 | 可解释范围 |
|---|---|---|---|
| milp_oracle | 完整未来实际日曲线 | 一次求整日并逐步执行 | 同一无损线性模型的完美预知参考；不是 AC 全局最优 |
| mpc | 当前观测持久性预测 | 默认 6 步滚动，每步只执行首动作 | 因果预测基线；未训练预测器 |
| MAPPO/IPPO | 当前公共信息、自身 SOC 和身份 | 策略动作经同一安全执行器 | 本版为共享网络的资源智能体 |

MPC 每个窗口末端均加目标 SOC 软罚，这是防止短视耗尽的终端启发式；整日真实计费只在真实日末加一次罚项。窗口截断不等同于 oracle 的完整目标。完美预知的信息优势和持久性预测误差不能解释为 RL 算法本身的优劣。

单独跑对照：

```bash
python -m vpp_mappo.baselines --config configs/ieee33_hourly.json --controller milp_oracle --csv data/opsd2019/test.csv --episodes 2 --output output/oracle
python -m vpp_mappo.baselines --config configs/ieee33_hourly.json --controller mpc --lookahead 6 --csv data/opsd2019/test.csv --episodes 2 --output output/mpc
```

compare 从检查点读取相同资源快照，自动顺序执行三组测试并输出 paired.csv、各组 trajectory.csv/episodes.csv/summary.json。非最优 MILP 可行解不能直接称为严格最优值，应检查 solver_optimal、solver_bound 与 solver_gap。

## 交流复核与尚未完成的研究

每步实际动作都调用 pandapower 交流潮流，记录最低/最高电压、线路负载率、网损、平衡节点功率、收敛与 AC 违规。ac_cost 使用含损耗的交流进口重新结算。线性 cost、AC cost 和两类违规分别报告。安全投影仅保证求解模型的约束，不保证交流可行；默认奖励仍基于线性模型，AC 结果是独立审计，不能称为闭环 AC 安全控制器。

暂未实现 AC 可行域修正、无功控制、可再生弃电决策、EV 到离站、DR 反弹、通信计算联合动作、DT/RAG/LLM 与多目标 Pareto。IEEE33 模式显式拒绝延迟/丢包/DT 开关，避免优化器和 RL 获得不公平状态信息；这些需要下一阶段统一部分可观测模型。当前已经具备真实曲线、标准网络、统一对照和审计路径，正式论文仍需充分训练、多种子、不确定性/超负荷压力和消融实验。
