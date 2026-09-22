# 第十一阶段：PCC口径、安全后备、OLS与真实DR

本阶段属于功能闭环验证，六项研究验收尚未全部完成。普通官方 MAPPO 保留独立训练、检查点与评价；新目标采用版本 v3，不覆盖旧版结果。

|任务|本轮实现与验证|未完成的研究验收|
|---|---|---|
|备用与安全口径|外层网损补偿+内层统一MILP；最终用AC PCC功率验收；成本与碳也改用含网损口径；30次激活、最大误差0.0954 kW|长预算重训；连续激活范围和任意未来扰动下安全保证|
|失败后备|常规MILP失败后有限现场候选搜索；硬约束和AC复核；EV/DR服务违约保留；无安全候选拒绝推进|真实总时限下的控制保障；极端故障应急供电/切负荷模型|
|Pareto基线|独立三维OLS角点与乐观LP优先级，调用官方固定权重MAPPO；4个子策略均完成训练与重载|统一总交互预算的长预算、多种子比较；收敛证据|
|DT收益|在v3下完成3个种子、物理/残差DT同观察MPC配对|成本差异CI跨零，收益未证实；真实漂移下覆盖仍需增强|
|真实EV/DR|因果提前离站导入/执行；CPUC/SCE A.4真实DR小时表完成解析和降额参数接口|完整真实EV数据尚缺；真实DR回补/成本参数不能从现表识别，降额尚未进入完整训练比较|
|工程与扩展|当前主机计时/能量接口探测、显式缺测标记|目标边缘设备与物理通信实测、真实能耗、IEEE69尚未完成|

## 目标版本与安全边界

- `linear`：v1，无损线性备用，旧实验保留。
- `ac_checked`：v2，对线性功率目标检查AC端点，仍不等于PCC实际跟踪。
- `pcc_checked`：v3，经济成本、碳排放使用AC并网点实际功率；备用目标通过PCC网损补偿和AC约束联合验收。训练与测试均启用现场AC保护及紧急后备。

PCC功率误差容差固定0.1 kW。备用确认检查比例 `[-1,-0.5,0,0.5,1]`，外层有限二分仅给出通过这些采样点的容量，不宣称连续全区间可行或最大备用容量。原线性包络只作为搜索上界提示。现场校核使用当前现场传感器，策略/critic仍只读公开观察。MILP内部仍为线性网络近似，最终由AC潮流验收，不能称为全局最优AC OPF。

紧急候选不调用MILP，但仍需AC潮流，计算耗时并非硬实时上界。可降级的是未来服务可达性，储能SOC、设备功率、DR能量/积压边界及网络约束仍检查。服务未达和违规计数不会归零；评价会将其列为不可行服务结果。没有通过检查的动作时停止仿真，不冒称现实系统已经安全停机。

## 实际实验

- 3个种子 × 5个激活比例 × 2种持续时间，共30次克隆回放；备用幅度固定20 kW。跟踪与后续EV/DR任务均通过，最大实际PCC误差0.0953949738 kW。这里只验证了上述幅度与样本。
- 普通MAPPO和偏好条件Pareto：各2个训练种子，每模型3回合×8步，共4个模型96个训练交互；各模型2个测试场景运行完成。
- OLS：1个种子，4个子策略，每子策略1回合×8步；训练32步，自适应验证64步，另有16步独立测试。普通/Pareto与OLS本次预算不相等，不能据此比较优劣。
- DT：3个预定合成种子，残差减物理DT的成本差均值约+0.000496，95%配对t区间约[-0.001405,0.002397]。两方法各24步的逐步九维联合覆盖均为100%，这不是漂移校准或真实数据覆盖的证明。保留原真实GB测试75%覆盖的未通过结论。
- 所有通过测试均为有限模拟样本；没有新增长预算收敛结论。

OLS的几何角点和优先级按线性支持定义独立实现，参考[MORL-Baselines LinearSupport](https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/linear_support/linear_support.py)和[MOMAland算法说明](https://momaland.farama.org/tutorials/learning_algorithms/)。我们的实现使用SciPy LP及三维角点枚举，未复制其JAX训练器。RL子问题并非精确最优解，故乐观差距不作为理论上界证书。所有子策略训练计入总量，自适应验证另行统计，下一轮比较必须包含这一预算。若角点耗尽提前停止，会显式记录未花完的预算。

修复了OLS传入NumPy偏好时检查点含NumPy标量而不能安全重载的问题：保存为Python列表，仍使用 `weights_only=True` 重载，没有放宽反序列化策略。调试失败目录单独保留在工作记录，不混入最终8个模型成果。

## 真实数据

[CPUC/SCE公开DR表](https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/energy-division/documents/demand-response/emergency-load-reduction-program/elrp-2022-program-data/sce-elrp-hourly-with-lip-values.xlsx)，来源页：[ELRP](https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-costs/demand-response-dr/emergency-load-reduction-program-data-and-information)。保存原始SHA256与派生JSON。

保留55条小时记录，其中35条有效激活、10条净响应为负。按 `Net MWh / (Nominated MW × 实际激活小时)` 计算响应比，不把非事件小时当成零响应。训练截止2022-09-04，校准为09-05至09-06，测试为09-07及以后。仅训练集得到的10%分位数约-0.0763，因此保守裁剪后的削减DR容量因子为0；这说明该简单经验规则不能支持正的可靠容量承诺，并非证明项目没有DR能力。

`conservative_flex`只生成削减DR容量/能量预算的降额敏感性参数，不改变平移、回补和成本假设。负值原始数据完整保留。表中时间未显式标区，按当地时间保留并标注假设，未与英国UTC曲线拼接。加州响应与英国能源曲线若组合，须标明跨来源实验。

[ACN官方EV数据](https://ev.caltech.edu/dataset)仍需完整JSON。公开网页下载此前返回截断JSON；API需用户自行注册获取访问权限，不应把凭据写进代码。找到的静态第三方仓库导出仅2字节，不能作为有效会话数据。现有导入严格要求首次决策前可见的申报电量，不拿最终交付量伪装需求。

提前离站事件通过 `--replay-actual-departures` 启用。规划仍用申报期限；真实离站只在事件到达后改变现场会话状态。提前离站未充电量只记一次。时间离散化假设写入导入审计，不能称为秒级事件回放。

## 运行

安装项目依赖；DR工作簿读取另需 `openpyxl==3.1.5`（已加入锁定文件）。

```bash
python -m unittest discover -s validation -p 'test_stage*.py' -v
python -m vpp_research.study_stage11 --dt-model PATH_TO_DT/residual.json --output runs/stage11
python -m vpp_research.dt_pcc_study --dt-folder PATH_TO_DT --output runs/dt_pcc.json
python -m vpp_research.ols --dt-model PATH_TO_DT/residual.json --output runs/ols --episodes 24 --policies 6 --reserve-mode pcc_checked
python -m vpp_research.dr_import --input sce-elrp-hourly-with-lip-values.xlsx --output runs/dr_events.json
python -m vpp_research.ev_import --input acn_sessions.json --output runs/ev.json --max-kw 7 --replay-actual-departures
python -m vpp_research.hardware_probe --output runs/hardware.json
```

OLS与现有campaign尚为两个入口；统一调度所有方法、匹配自适应选择交互、长预算和统计报告是下一阶段任务。当前硬件探测没有能量计数器，不能把模型能耗当成设备实测。IEEE69尚未接入，现网络类仍具有IEEE33假设，不能只改名称冒称扩展完成。

最终回归：98项测试通过（36.043秒），覆盖PCC实际执行、两步激活、后备服务降级、拒绝执行、OLS角点/预算、提前离站因果性及真实DR降额规则。
