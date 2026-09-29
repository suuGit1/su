# v4 运行与实验说明

本版本针对 v3 图表中出现的储能提前耗尽、候选/执行脱节、训练不易诊断等问题修改。用户原始 120 回合逐步轨迹未提供，因此不能声称全部根因已确认，更不能预先保证性能提升。

## 已实施修改

- 观察从65维扩展为80维，加入已接收候选、最新待执行命令、预计到达时间、队列长度和时序开关；不输入零时延现场 SOC/执行功率。该接口仍是部分观察，不声称完整 Markov 状态。
- 储能候选范围由公开DT区间和待执行命令预测；只是估计范围，实际安全仍由统一执行器验收。所有学习基线使用同一动作映射。
- 原始动作零均值从低弃电起点探索；通信/计算在主要探索区间使用平滑正分配，将截零阈值移至原始动作-2，保留零分配/关闭上传能力。
- Pareto actor/critic 独立优化和梯度裁剪；PopArt 不改变三目标原始单位；新增 KL、熵、裁剪比例、梯度及分目标价值误差诊断。
- 多个完整日组成一批 PPO 更新，GAE 在每个终止处截断，尾批不补造交互；训练日期按种子确定地洗牌。
- 固定验证日期/偏好生成学习曲线，并选定检查点；验证调用额外计费，不读取测试结果选模型。
- 严格HV不变；新增全日费用分项、终端惩罚、弃电、备用未确认及无效ESS请求统计。

普通 MAPPO 仍调用 vendor/mappo 官方更新器，经济单目标独立保留；central、fixed、OLS、MPC、MILP oracle 全部保留。v3 入口可继续使用，v3 检查点不能当作80维v4模型续训。

三目标仍为经济—碳—备用，尺度 [100,100,100]。经济目标含AC运行费用、通信计算费用和终端惩罚；备用是同一时段起点的有限AC激活点确认，不是瞬态频率性能或完整AC鲁棒安全保证。

## 安装与运行

可沿用已跑通 v3 的虚拟环境。新建环境建议 Python 3.11，依赖见 requirements-research-lock.txt。v4 研究入口当前只验证CPU，不自动启用RTX4060。

```powershell
python -m pip install -r requirements-research-lock.txt
```

在项目根目录执行，run_v4.py 自动启用UTF-8；直接运行其他脚本建议 `python -X utf8`。

先做全日功能验收：

```powershell
python run_v4.py --data-root data/real --methods mpc ordinary pareto --seeds 1 --episodes 4 --rollout-episodes 2 --validation-every 4 --eval-days 1 --selection-days 1 --dt-train-days 2 --dt-calibration-days 9 --workers 1 --output runs/v4_smoke
```

这只验证链路，不用于论文比较。

中等预算诊断：

```powershell
python run_v4.py --data-root data/real --methods mpc ordinary pareto --seeds 1,2,3 --episodes 120 --rollout-episodes 8 --validation-every 40 --eval-days 3 --selection-days 2 --workers 1 --output runs/v4_diagnostic_120
```

先检查固定验证曲线、终端惩罚、SOC、弃电、动作失效与可行率，再决定是否扩大。现有29天已用于v3问题诊断，正式论文最好增加新的保留测试时段；不要用测试数据反复调参。

七方法长预算示例：

```powershell
python run_v4.py --data-root data/real --seeds 1,2,3,4,5 --episodes 600 --rollout-episodes 8 --validation-every 120 --eval-days 29 --selection-days 2 --workers 1 --output runs/v4_full_600
```

600是可调整的预算示例，不是保证收敛的阈值。固定权重/OLS子策略合计使用总回合上限；OLS仍可能早停，不能隐瞒或称所有运行严格等预算。MPC/oracle不训练，跨训练种子的相同结果不构成独立学习实验。

还支持 --learning-rate、--ppo-epochs、--target-kl、--cpu-threads、--solver-time-limit、--eval-preferences、--dt-train-days、--dt-calibration-days、--ols-policies、--hv-reference、--ev-sessions、--dr-mode。

## 验证、选择与恢复

验证在指定回合间隔后的完整更新边界执行，末批也验证；每次4个固定偏好×selection-days。普通/central在优先验证可行数后比较经济效用，fixed比较对应加权效用，Pareto比较HV。未启用定期验证时使用最终检查点。OLS保留其子策略验证选择流程。

--validation-every 0 可关闭定期验证，但最终策略选择仍保留。验证调用、重试和测试调用分别报告，不能只按训练回合数声称开发总交互相同。

恢复使用完全相同命令加 --resume；更改数据、配置或预算必须换输出目录。恢复点为完整更新批次；中断批次重做会增加实际调用。未知完成状态的交互拒绝自动恢复。已保存的功能验收模型不代表正式模型。

## 报告

```powershell
python -X utf8 scripts/report_v4.py runs/v4_diagnostic_120
```

- report.json / summary.csv：原严格HV与经验IGD。
- v4_report.json：失败分类、验证曲线、实际验证预算与审计。
- daily_metrics.csv：全部有记录的测试案例，明确标注不可行；费用分解、弃电、SOC、备用确认和动作偏差。
- training.csv：训练日回报、终端项及更新诊断。
- validation_curve.json：固定验证结果。
- best_validation.pt / latest.pt / resume.pt：验证选定、最终和恢复检查点。
- figures/v4_validation_learning.*：固定验证HV和可行率。
- figures/v4_cost_components.*：运行费用、终端惩罚和弃电。
- figures/v4_feasibility_action_gap.*：可行率与有请求但ESS执行为零的比例。

同日不同偏好及跨种子重复日期存在关联，不能把所有点当作独立样本做显著性检验。证据完整、无物理违例、区间证书成立和备用确认是不同指标。

## 数据与范围

沿用 data/real 中 OPSD GB 能源/价格曲线、NESO碳数据及来源记录；缺失数据不静默回退合成曲线。国家曲线缩放驱动IEEE33仿真，不是IEEE33实测。真实完整EV会话缺失时默认关闭EV，DR默认关闭；SCE参数仅为显式跨地区敏感性。没有新增实测硬件时延/能耗证据，也不声称解决了这些数据缺口。

## 回归

```powershell
python -X utf8 -m unittest validation.test_v3 validation.test_v4 -v
```

正式性能需在重新训练后确认。不能通过删除失败种子、更换测试日期或放松安全口径制造优势。
