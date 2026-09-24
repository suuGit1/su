# 三条连接与统一实验入口

本版在真实数据入口默认启用 `configs/connected_ieee33.json`。普通 MAPPO 独立保留。

```bash
python -m pip install -r requirements-research-lock.txt
python scripts/run_all_seeds.py --methods mpc milp_oracle ordinary central fixed ols pareto --seeds 1,2,3 --episodes 100 --eval-days 29 --selection-days 2 --ols-policies 4 --output runs/connected_100
```

先用 `--dry-run` 检查日期与预算；同目录续训加 `--resume`。`--episodes` 是每种方法、每个种子的总训练回合；固定权重三策略、OLS 所有子策略合计使用这个预算，余数分配至前几个子策略。OLS 没有未访问角点或验证失败时会提前停止，`budget_complete=false`，不宣称完成等预算实验。策略选择交互独立计数，因此这里匹配的是训练预算，不是开发总交互预算。当前命令是预算配置示例，不代表已经完成长预算训练。

## 已接通的行为

|连接|实际作用|边界|
|---|---|---|
|DT区间→安全|区间盒约束投影；最终物理执行动作再次检查证书|只证明单步线性约束；AC校验单列，不是AC鲁棒保证|
|计算/下行→闭环|控制计算占用CPU并计能耗，命令进入带截止期队列；缺命令交给现场安全执行器|调度步边界模型，正延迟通常在下一步执行；不是毫秒级硬件仿真|
|任务/风险→协调|偏好、任务类型与区间宽度改变带宽/CPU优先级和预留比例|规则协调器，风险分数不是校准概率；不在回合内偷偷改变奖励偏好|

`interval_safety`、`command_timing`、`risk_adaptive` 可分别关闭作消融；`task_focus` 支持 balanced/economic/carbon/reserve。下行带宽、传播延迟、命令截止步数、控制计算周期在配置文件中定义。DT初始离线采集禁用尚未校准的区间保护，保留其他闭环配置。计算时间来自显式周期假设，不能当作设备实测。

## 基线与报告

|方法|定义|
|---|---|
|ordinary|官方MAPPO组件，经济奖励，不被Pareto版本替换|
|pareto|偏好条件actor、向量critic/GAE|
|central|集中式PPO，经济目标|
|fixed|三个固定权重MAPPO，独立验证选择子策略|
|ols|OLS选择权重，官方MAPPO求近似子问题；优先级不是理论上界证书|
|mpc|受限公开观察上的经济MPC，与策略使用同一执行器|
|milp_oracle|完美预知曲线/会话的滚动MILP，单列为非因果参考；不是闭环成本严格下界|

验证集按日期前段用于DT校准（默认12天），后续独立日期用于策略选择（默认2天）。训练、DT校准、策略选择与测试的用途和日期写入 manifest；不能用测试收益选子策略。DT拟合使用训练集。真实测试当前最多29个完整日期，不重复日期凑数。

输出 `results.json`、`report.json`、`summary.csv`、`report.md`。重建报告：

```bash
python -m vpp_research.real_report runs/connected_100
```

经济/碳/备用向量固定为 `[-cost,-carbon_kg,reserve_kwh]/100`，成本包含终端惩罚，成本与碳采用AC并网点核算。备用采用含网损PCC采样核验；未确认步骤不得混入有效Pareto前沿。HV参考点默认预先固定 `[-100,-200,0]`，可用 `--hv-reference` 在运行前修改；IGD参考集来自独立验证前沿，不是真实最优前沿，跨日期比较也包含场景差异。没有可行前沿时 HV=0、IGD=null，不能解释为已验证性能优势。

## 尚未成立的研究结论

小预算连接验证发现：2天DT拟合、9天场景块校准得到很宽区间，区间安全问题不可行；现场AC执行仍可通过。记录 `guard_feasible`、`guard_inconsistent_dimensions`、最终 `guard_certificate_survived`，不会通过缩小区间掩盖失败。下一步应改善DT拟合代表性和区间有效性，再开展长预算对比。

真实EV完整会话仍缺失，默认明确关闭；同地区DR参数仍不足，SCE只能显式用作跨地区敏感性来源。真实能源曲线是英国国家曲线缩放到VPP，并非IEEE33实测。现阶段不要把接口齐全等同于全部论文证据完成。

## 本次实际验证

14项针对性检查通过。真实数据种子82，每种学习方法计划4回合、每回合24步；DT拟合2天、校准9天，策略选择1天，测试1天，一个测试偏好。普通/Pareto/集中式/固定权重各完成96个训练步。OLS首个子策略验证失败，完成24/96步后停止，未伪装成等预算完成。

普通、Pareto、集中式、MPC、MILP参考各完成24步测试，基础约束与AC违规均为0，但备用未确认步骤分别为4、1、10、1、3；区间保护均不可行24步。固定权重与OLS均无合格策略可供选择，未开展其测试。因而本次所有方法的有效前沿为空，HV=0、IGD未定义，不支持方法优劣结论。

原始协议、结果及统一报告见 [experiments/connected_system](../experiments/connected_system)。下一步先诊断宽区间与PCC备用核验失败，再补真实EV/DR并扩大预算。
