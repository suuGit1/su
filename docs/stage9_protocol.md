# 第九阶段：机制验收、公开数据与预算扩展

本阶段把研究清单转成可执行实验，但程序通过测试不意味着所有论文假设通过验收。第八阶段的三目标协议保持版本不变，备用仍明确是线性下一步包络，不能升级措辞为 AC 安全的实际履约能力。

## 实验入口

使用仓库现有依赖，以下命令从仓库根目录执行；`runs/dt` 可由第八阶段 DT 命令生成。

```bash
python -m vpp_research.mechanisms --dt-folder runs/dt --output runs/stage9/mechanisms.json
python -m vpp_research.reserve_activation --dt-model runs/dt/residual.json --output runs/stage9/reserve.json
python -m vpp_research.campaign --dt-folder runs/dt --output runs/stage9/campaign --episodes 60
python -m vpp_research.timing_study --model runs/dt/residual.json --output runs/stage9/timing.json
python -m vpp_research.public_data --opsd data/raw/opsd.csv --carbon-folder data/raw/gb_carbon --output data/gb_profiles
python -m vpp_research.gb_study --profiles data/gb_profiles --output runs/stage9/gb_study
python -m vpp_research.ev_import --input data/raw/acn.json --output data/acn_sessions.json --max-kw 7
python -m unittest discover -s validation
```

`campaign` 保留普通官方 MAPPO，加入联合公开观察的集中式 PPO。它使用联合动作对数概率之和及经济标量价值；不是把分散 actor 改名为集中式算法。Pareto 四种 C3 模式都保留 18 个动作维度，固定未启用的通信/计算分量，不改变网络规模。四组均关闭协调器，另以 `coordinator` 与 `joint` 对比；`no_residual` 使用物理 DT。固定权重三模型分享总训练预算。

默认 60 回合是预算扩展验证，不自动等于收敛。可改成 600、1200 等预注册预算，按验证学习曲线决定正式预算；不能按测试表现选择模型。每个完整实验单元单独保存，重复运行跳过已完成单元；中途未完成的训练目录需要保留并改用新输出目录，当前不是优化器状态级断点续训。配置或 DT 内容改变时拒绝复用实验目录。

测试使用 15 个预定内部偏好和 3 个独立场景，验证使用顶点和均匀偏好。IGD 参考集仅来自验证点，所有失败回合保留。不同输出目录必须先合并验证参考集才可比较 IGD；不能直接比较各目录独立计算的 IGD。

## 机制与备用审计

七种扰动在运行前固定：正常、上行慢、计算慢、丢包、负荷增大、风电增大、组合扰动。每条件 8 个独立种子，保持同信息 MPC、现场 AC 执行校核和费用定义。记录候选动作与执行动作，检验安全投影是否削弱 DT 差异。多个条件的置信区间是探索性统计，没有进行多重比较校正。

备用审计分两层：线性极值动作的预测/实际 AC 检查；指定净购电目标的完整克隆轨迹执行。新 MILP `grid_target` 约束显式固定第一步净购电。上/下调各自独立激活一个调度步，随后按当前信息规划直到回合结束，检查后续约束和任务。该审计可否定旧备用申报，并不自动形成新的安全备用定义。若失败，不能删除样本或仍宣称旧指标已经履约。

## 下行、计算与后备

`ResearchEnv` 的可选 `timing_contract` 加入命令到达/过期队列。延迟非零的命令仅在随后调度边界应用；没有可用命令时调用同约束的现场经济规划，仍不可行则保留失败。控制任务按显式周期假设占用 CPU 预算，剩余平均 CPU 速率供 DT 队列，控制能耗进入费用/碳核算。这是按步长平均的资源模型，尚非细粒度实时操作系统仿真。

推理与安全求解的主机墙钟时间单独统计；不能视为已在用户边缘设备测得的时延。默认数据步长 15 分钟，不能支持继电保护、频率动态或毫秒级实时控制结论。物理容量不足时的紧急负荷/EV 服务降级尚未实现，不能把仿真停止当设备后备保障。

## 真实数据及因果边界

- [OPSD 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06/)：使用 `GB_GBN` 的负荷、光伏、风电和日前价格。原价格单位 GBP/MWh，转换为 GBP/kWh；其他费用参数声明为 GBP 假设，不使用隐含汇率。国家曲线按一月训练峰值缩放至 IEEE33。
- [NESO Carbon Intensity API](https://carbon-intensity.github.io/api-definitions/)：UTC 半小时实际碳强度，两个完整区间算一小时均值，不插值缺值。2019 年一月训练、二月校准、三月测试。记录每个原始响应 URL 和 SHA256。
- 碳实际值用于事后核算；策略输入使用上一小时实际值的滞后代理，再经遥测到达。未取得历史发布时间档案，因而这是明确的信息可用性假设，不是已核验的历史在线回放。
- [ACN 官方会话](https://ev.caltech.edu/dataset)：导入器核对 `modifiedAt`，禁止将后续申报或最终 `kWhDelivered` 作为到站需求。桩容量必须显式提供；申报离站与实际离站分开记录。目前实际提前离站尚未进入在线动态回放。英国数据实验关闭 EV，不把美国会话拼接成英国实测。

## 尚不能宣称已完成的项目

DT 显著改善控制、Pareto 训练收敛与统计优势、AC 备用安全定义、独立分布漂移下的可靠校准、真实 EV 提前离站、真实 DR 参数、现场 CPU/网络校准、紧急服务降级和递归可行性都需要独立验收。它们不是增加几个配置文件就能完成的结论。

## 延迟标签再校准

`python -m vpp_research.recalibration --dataset runs/dt/datasets.json --model runs/dt/residual.json --output runs/stage9/recalibration.json` 按场景顺序预测，完整场景标签至少延迟一个场景边界后才进入滚动窗口。它是因果离线回放；尚未接入完整真实标签到达队列，不声称分布漂移下恢复严格覆盖保证。

## AC 备用修正候选

`python -m vpp_research.checked_reserve --model runs/dt/residual.json --output runs/stage9/checked_reserve.json` 在当前现场真实状态下，对给定基线的上、下端点求共享约束 MILP 并校核 AC；端点失败则有限次二分缩小容量，基线不可行则零申报并明确标记。后续轨迹使用 AC 校核经济规划。它没有改变旧目标，也没有把旧策略当成按新备用目标训练；若正式采用，必须发布新目标版本并重训全部对照。与旧激活试验同时改变了容量确认和后续 AC 规划，不能把改善全归因于其中一个组件。

真实英国训练发现 L1 投影在某状态被预处理误报不可行，而同约束经济规划有可行解。修复为仅在剩余总求解预算内关闭预处理复核，仍执行原独立约束残差验收；没有放松边界。回归用例保存了该状态。

## 新备用目标版本的独立入口

`campaign --reserve-mode ac_checked` 使用 `c3-cost-carbon-ac-endpoint-reserve-v2`，将现场双端点 AC 确认容量作为备用分量。它保留旧 `linear` 入口和普通 MAPPO，保存并校验目标版本，禁止把新旧前沿直接混比。新版本仅对当前工况的两个端点做有限求解确认；没有递归鲁棒保证。第九阶段的五种子主比较仍为旧版本，新版另做训练/重载短程检查，正式采用前需要按同预算重训完整对照。

## 功率计量与比较范围

新旧备用量都沿用线性净购电的 kW 口径，AC 校核用于验证电压、线路与潮流收敛；它不等价于含线路损耗的 PCC 实测功率跟踪。正式市场备用履约还需校正网损和计量口径。`ac_carbon_kg` 单独提供 AC 根节点功率的排放诊断，不改写原有训练目标。

本轮九方法族训练与评估的运行时间受共享环境并发负载影响，交互预算匹配不能解释为墙钟资源严格公平；硬件时延结论应使用隔离的设备基准。
