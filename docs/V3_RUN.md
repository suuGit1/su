# v3完整实验版

Digital-Twin-Driven Communication–Computation–Control Co-Design for Safe Pareto Multi-Agent Energy Management in Cyber-Physical Virtual Power Plants。

本版完成A1训练审计、A2实验协议、A3逐步证据与统一报告、B1偏好响应诊断。IEEE33、规则任务/风险协调器、C3角色、残差DT、区间安全层与现场AC执行器、独立普通MAPPO及Pareto-MAPPO全部保留。没有将B2之后的长预算、真实EV/DR和工程验证宣称完成。

## 安装

从仓库 `release/v3` 分支获取完整代码。建议Python 3.11、独立CPU环境，不混用历史CUDA11.3依赖。模型协议仍只支持本研究CPU入口。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research-lock.txt
```

Windows激活命令为 `.venv\Scripts\activate`。训练所需的真实预处理日曲线已在 `data/real` 中；缺文件或协议错误会报错，不回退合成曲线。

## 统一运行

```bash
python run_v3.py --seeds 1,2,3 --episodes 12 --eval-days 3 --selection-days 1 --workers 3 --output runs/v3
python scripts/report_v3.py runs/v3
```

七种方法为 `mpc milp_oracle ordinary central fixed ols pareto`。固定权重与OLS子策略共同分配每方法、每种子的总回合预算。OLS无新角点允许提前停止，剩余预算明确保留为未使用。

仅双MAPPO：

```bash
python run_v3.py --methods ordinary pareto --seeds 1,2,3 --episodes 100 --eval-days 29 --selection-days 2 --workers 3 --output runs/v3_dual_100
```

100回合只是可运行的配置示例，不表示已经完成。其他开放参数继承v2：学习率、PPO轮次、求解时间限制、DT拟合/校准天数、偏好、OLS子策略数、EV会话和DR模式。使用 `python run_v3.py --help` 查看。

`--dry-run`只预检并冻结清单。对同一目录执行实际运行或恢复需加 `--resume`。改变预算、日期、数据、核心代码或约束后使用新目录，不绕过协议检查。训练恢复仍受原始绝对路径约束，跨机器自动迁移训练目录未实现。

## A1：交互和恢复审计

- `interactions.sqlite`以同步事务记录每次训练环境调用开始、成功或异常；进程突然退出留下的pending调用视为结果未知，禁止自动续训。
- 每次完整PPO更新后检查点保存回合与attempt关联；账本核对该回合全部step后提交。重做回合计入实际调用，但不会重复算作已提交训练步。
- `attempts.jsonl`和训练轨迹同步落盘；审计核对事务记录的完整step标识序列，不能只比较总数。
- 检查点已保存但回合日志提交前中断，可依据检查点关联修复账本提交关系；不补造缺失轨迹。只有最终检查点未生成时，可恢复已完成全部回合的resume.pt以导出模型。
- v2存在名义4176步与日志4149步差异，且部分轨迹也缺项；现有证据不能定位历史数据缺失的根本原因，不能补记为“已证实成功”。v2检查点仍可推理，训练恢复须新建v3实验。

```bash
python -m vpp_research.ledger runs/v3 --output runs/v3/training_audit.json
```

`training_steps`为已完成名义回合步数；`actual_training_steps`在v3为账本成功调用数；`attempted_training_calls`包含失败或未知调用。重试可能让实际调用超过名义预算。验证/测试有独立的 `evaluation.sqlite`，报告累计所有尝试（包括重试归档）的调用；不能漏掉选择成本后声称开发总交互等预算。

## A2：不可混用的实验口径

清单中的 `experiment_contract`保存核心源码及官方MAPPO源码摘要、资源文件哈希、真实数据指纹、观察/DT/目标版本、固定尺度、HV参考点、日期划分和各算法信息条件。完整清单比较还检查配置、方法与预算。

三目标向量为 `[-经济成本,-碳排放,+备用能量]/[100,100,100]`。经济成本含AC网损对应并网成本、通信计算成本与终端惩罚；碳排为进口电力及通信计算能耗，不计出口减排信用；备用为同一时段起点下有限激活点AC核验的对称PCC备用能量。不能把它解释成额外执行后的备用或频率动态响应。

MPC使用受限公开DT观察，预知MILP明确为非因果参考。学习方法与控制基线使用统一执行约束，但MPC固定通信计算候选与学习型C3分配不同，属于系统方案对比；纯算法因果比较仍需后续受控消融。

`strict_training_budget_hv`只纳入事务审计通过、名义预算完成、尝试调用与成功调用均等于训练上限的学习运行；选择/测试费用仍另外列示。原 `budget_matched_hv`兼容字段只筛选名义回合预算，不得作为严格等实际交互证据。

## A3：结果、轨迹与图表

每种学习方法的每个验证与测试场景保存 `trajectory.jsonl`、`evaluation.json`及评价事务账本；控制器保存相同轨迹和 `summary.json`。失败保留部分轨迹与异常，重试归档旧目录。

四阶段能量命令为发出、下行接收、区间修正、最终执行；还保存协调决策、带宽、CPU、AoI、DT观察/区间、AC指标和备用核验。`energy_candidate`历史字段含现场上游处理，分析原始策略指令应使用 `issued_energy_candidate`。

统一报告入口：

```bash
python scripts/report_v3.py runs/v3
```

输出：`report.json/md`、`summary.csv`、`training_audit.json`、`trace_audit.json`、`v3_report.json/md`和 `figures/`。图表含HV、可行非支配点、三目标、安全认证/违反、名义与实际训练预算、训练回报及各方法典型日调度。若有DT专项结果也生成其对照图。

`evidence_complete`要求训练账本/日志与验证/测试轨迹复算均通过；它不表示全部区间认证、算法优越或收敛。图中的前沿点不是独立统计样本；训练偏好和日期变化时的回报曲线也不是固定偏好收敛证据。更细的节点电压空间图需另外导出节点级潮流结果。

## B1：偏好与安全执行诊断

诊断使用冻结模型，在相同初始观测、物理状态和环境随机状态上改变偏好，随后持续推进默认2个时段，比较发出/接收/区间修正/最终执行的差异，以及向量critic输出、tanh饱和比例、区间修正量和安全指标。

```bash
python -m vpp_research.diagnose_preferences --checkpoint runs/v3/seed_1/pareto_1/latest.pt --csv runs/v3/seed_1/selection/validation.csv --days 1 --states 3 --output runs/diagnose_pareto
python -m vpp_research.diagnose_preferences --checkpoint runs/v3/seed_1/ordinary_1/latest.pt --csv runs/v3/seed_1/selection/validation.csv --days 1 --states 3 --output runs/diagnose_ordinary
```

输出 `diagnostics.json`、当步偏好响应和跨时段响应PNG/SVG。`--probe-steps`控制反事实窗口，诊断点避开无法完成窗口的日末时段。普通MAPPO不接收显式偏好，作为策略响应负对照；执行器仍可能响应偏好。执行动作无差异可能来自下行队列、区间投影或现场约束，不能一概归因安全层。该短窗口诊断不替代完整回报的未见偏好评估，也不自动修改算法以追求优势。

## 验证与边界

```bash
python -m unittest discover -s validation -p "test_*.py"
```

重点验证未知交互拒绝恢复、重试记账、回合缺步拒绝提交、轨迹复算、向量GAE终止掩码、actor偏好接口及双MAPPO恢复一致性。已完成实验的确切配置与结果见 `experiments/v3/`；开发验收不作为长期性能结论。

真实EV完整会话、同地区DR、C3硬件测量和大网络证据仍需后续补充。当前框架是国家真实曲线缩放驱动的IEEE33仿真，不是现场部署。
