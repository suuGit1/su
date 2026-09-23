# 集成运行版 v1（合成演示历史说明）

**当前默认入口已改为真实数据。请优先阅读 [真实数据开放命令行](REAL_DATA_CLI.md)。下面的旧版合成演示命令均须增加 `--synthetic`。**

本版本将 IEEE33、任务/风险协调器、能源/通信/计算智能体、混合 DT、安全执行器和双 MAPPO 接入同一个运行入口。默认数据均为明确标注的合成示例，不需要下载外部数据即可完成闭环。

## 安装与第一次运行

建议 Python 3.11，在项目根目录执行。Windows 创建虚拟环境后使用 `.venv\Scripts\activate`；Linux/macOS 使用 `source .venv/bin/activate`。

```bash
python -m venv .venv
# 请先执行上方对应操作系统的虚拟环境激活命令
python -m pip install -r requirements-research-lock.txt
python run_vpp.py all --output runs/integrated
```

安装前请先激活虚拟环境。项目已包含固定版本的官方 MAPPO 核心代码，无需再次下载 on-policy。默认使用 CPU、8 步时域、两回合训练；执行每一步包含 MILP 与交流潮流校核，耗时高于普通 Gym 示例。不要将默认预算用于论文性能结论。

`all` 按顺序完成：独立场景 DT 数据采集 → 残差训练及区间校准 → MPC 闭环 → 普通 MAPPO 训练及独立闭环 → Pareto-MAPPO 训练及未见偏好闭环 → 自动架构验收。

成功标志为根目录 `summary.json` 的 `status=completed` 和 `acceptance.json` 的 `passed=true`。未完成、失败或不存在汇总文件均不能视为通过。

## 分开运行各模块

```bash
python run_vpp.py prepare --output runs/dt
python run_vpp.py run --method mpc --dt-model runs/dt/residual.json --output runs/mpc
python run_vpp.py train --method ordinary --dt-model runs/dt/residual.json --episodes 20 --output runs/ordinary
python run_vpp.py train --method pareto --dt-model runs/dt/residual.json --episodes 20 --output runs/pareto
python run_vpp.py run --method ordinary --checkpoint runs/ordinary/latest.pt --output runs/ordinary_test
python run_vpp.py run --method pareto --checkpoint runs/pareto/latest.pt --preference 0.2 0.3 0.5 --output runs/pareto_test
```

普通 MAPPO 始终使用经济奖励，不会因测试请求偏好改变为 Pareto 策略。Pareto actor 使用偏好输入，critic 为三维向量，训练使用向量 GAE。两者使用相同物理系统、C3 角色、协调规则和现场安全执行器。

## 回合边界断点续训

```bash
python run_vpp.py train --method pareto --dt-model runs/dt/residual.json --episodes 40 --resume --output runs/pareto
```

`--episodes` 表示目标总回合数，不是新增回合数。普通 MAPPO 使用相同命令，修改方法和目录即可。仅允许增加目标回合数，算法、数据、DT、资源参数等不一致时拒绝恢复。

每次完整 PPO 更新后原子保存 `resume.pt`，包括策略/价值网络、优化器、归一化状态、偏好采样器及 Python/NumPy/Torch 随机状态。`latest.pt` 是本次训练成功结束时生成的推理模型；中途中断后应先续训，再使用最新推理模型。

不支持从半个回合的物理状态恢复；中断回合重做。`attempts.jsonl` 记录执行尝试和已记录交互，`trajectory.jsonl` 追加保留尝试标识，`training.csv` 只统计完成更新的逻辑回合。中断重做的开销不能混入逻辑预算而被忽略。进程被强制杀死到来不及写日志时，最后一步是否完成可能无法确认；正式预算实验需把这种执行标为中断。

## 每个时间步的闭环

1. DT 根据历史到达遥测和已发命令产生估计，残差模型校正估计并附加九维区间与 OOD 标识。
2. 能源六角色产生 ESS、EV、DR 转移、DR 削减、PV/风电弃电候选动作；通信六角色和计算六角色产生资源分配权重。
3. 协调器仅读取公开观察，生成刷新/服务任务、风险分数和保底资源分配；与策略资源请求混合。
4. 现场安全执行器读取本地传感状态，求解统一约束 MILP 并进行 AC 验收；失败时尝试紧急候选，仍无硬安全解则停止并报告失败。
5. 物理 VPP 执行动作，通信/计算队列推进，已完成遥测更新 DT。下一步策略不能直接访问未到达的现场真值。

协调器是规则式，不是 LLM。偏好是用户任务输入（推理）或采样任务（训练）；风险评分是阈值启发式，不是违约概率。通信/计算动作改变队列、AoI 和 DT 更新，能耗采用显式模型参数。

## 输出与目标口径

|文件|含义|
|---|---|
|`dt/assessment.json`|DT 训练/校准/测试场景划分、RMSE 和覆盖率|
|`ordinary/latest.pt`、`pareto/latest.pt`|两套独立推理模型，包含 DT 与资源参数快照|
|`*/resume.pt`|完整回合更新后的续训状态|
|`*/trajectory.jsonl`|公开观察、DT 估计和区间、策略动作、协调任务、实际 BW/CPU、执行动作、反馈与安全记录|
|`*/summary.json`|成本、碳、备用、越限、服务缺口、丢包及 DT 更新统计|
|`acceptance.json`|三个模式的闭环结构与本次场景安全检查|

三目标固定为成本最小、碳最小、对称备用最大，学习向量为 `[-cost,-carbon,+reserve]/[100,100,100]`。成本含网损、通信计算模型费用及终端罚项；碳按 PCC 实际功率核算。备用采用有限采样的 PCC 跟踪确认，终端备用为零；不能将它解释为连续范围或无限时域安全证明。汇总的备用单位为 kWh，是各步指标之和，不是瞬时额定 kW。

## 异常场景验收

```bash
python run_vpp.py run --dt-model runs/dt/residual.json --fault packet_loss --output runs/fault_packet
python run_vpp.py run --dt-model runs/dt/residual.json --fault solver_timeout --output runs/fault_solver
python run_vpp.py run --dt-model runs/dt/residual.json --fault early_departure --output runs/fault_ev
python -m unittest discover -s validation -p 'test_*.py'
```

丢包场景设上行丢包率为 100%；求解超时是在首步现场规划器注入失败，用于验证紧急控制调用链，不是操作系统级实时期限证明；提前离站为合成事件，必须保留未满足需求，不能为了通过安全测试抹去服务缺口。

## 模块状态与后续扩展

|模块|当前状态|边界|
|---|---|---|
|IEEE33、ESS/EV/DR/PV/WT|已接入统一闭环|默认场景与资源布点为仿真假设|
|协调器、18 个 C3 角色|已接入训练与推理|协调器为规则式；18 角色分为能源、通信、计算三组|
|DT 物理预测/残差/校准|已接入数据准备与在线估计|OOD 可记录；不自动在线重训或保证漂移覆盖|
|安全执行器|统一 MILP＋AC＋紧急后备|未实现独立 QP 后端；不能保证任意故障均可安全服务|
|普通/Pareto MAPPO|独立训练、推理、恢复及轨迹|默认短训练不证明算法优势|
|真实数据|保留现有 OPSD/GB、EV、DR 导入模块|真实实验需单独配置场景与会话；本入口的 DT 准备是合成演示流程|
|硬件与通信|事件模型、本机计时能力保留|无线设备与硬件能耗未实测|

后续工作应围绕这一入口增加真实数据准备、长期实验配置、自动再校准和实验矩阵，不再要求用户拼接各阶段脚本。
