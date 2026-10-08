# v4.1：P0-1～P0-4 交付说明

本版本保留普通官方 MAPPO、Pareto-MAPPO、集中式 PPO、固定权重、OLS、MPC 与 MILP oracle。经济—碳—备用尺度和严格 HV 口径不变。代码可运行不等于已证明论文性能优势。

## 四项修改

| 任务 | 实现 | 边界 |
|---|---|---|
| P0-1 备用失败 | v4.1 对备用包络等 MILP 不可行结果在剩余时间内关闭预处理复核；每个解仍做独立残差检查。记录同周期初始SOC、并网点基线、包络解状态、重试和激活点失败 | 复核仍无解或不最优时保持未确认，不把资源不足计为成功；备用仍为有限AC激活点确认 |
| P0-2 区间安全层 | 记录冲突顶点、电压/支路/并网点约束及诊断松弛；可行区间约束直接加入现场MILP，执行后复核原证书 | 不缩窄校准区间、不执行松弛解；证书仅覆盖单步线性网络/SOC/DR约束，不是连续AC鲁棒或EV期限证明 |
| P0-3 后备与降级 | 现场重求解共享时间预算；超时、不可行、AC失效明确分类；无MILP有限候选后备独立检查硬约束和AC；记录EV/DR不可达量 | 无已验证硬可行动作则拒绝推进。服务降级与区间未认证显式计数，不宣称后备一定存在或已满足现场实时性 |
| P0-4 延迟感知MPC | 读取80维公开观察，结合自身发令历史，按实际调度边界预测新命令到达状态；一次消费命令，处理断链、过期和终端后到达 | 使用持久状态预测，现场安全修正未知；中途接管且队列不完整时标记。通信/计算仍固定，因此联合C3比较是系统方案比较 |

默认 `run_v41.py` 使用 `configs/v41_ieee33.json`，其中 `safety_revision=1`。`run_v4.py` 保持旧配置默认值0。新训练必须用新输出目录，旧检查点不能通过修改配置无声续训；专门的候选重放工具允许显式选择修订版本。

## 运行

沿用 v4 环境，或使用 Python 3.11 安装 `requirements-research-lock.txt`。本版本验收使用 CPU。

```powershell
python -X utf8 -m unittest validation.test_v3 validation.test_v4 validation.test_v41 -v
python run_v41.py --data-root data/real --methods mpc ordinary pareto --seeds 1 --episodes 2 --rollout-episodes 2 --validation-every 2 --eval-days 1 --selection-days 1 --dt-train-days 2 --dt-calibration-days 9 --eval-preferences "0.2,0.3,0.5" --workers 1 --output runs/v41_smoke
python -X utf8 scripts/report_v41.py runs/v41_smoke
```

上述是单种子小预算验收，不适合论文性能结论。训练、验证和测试调用分别计费。断点恢复使用原参数加 `--resume`。

七方法短链路仍可用 `scripts/prepare_v3_acceptance_data.py`，至少4回合以覆盖固定权重和OLS的子策略预算。不要用两小时截断结果替代24小时调度证据。

## 诊断与复现

- `v41_safety_report.json`：各方法区间冲突原因、现场安全模式、备用状态、预处理重试、服务降级。
- `trajectory_complete.jsonl` 中 `feedback.reserve_diagnostic`：同周期备用确认的求解与物理状态。
- `feedback.guard_conflicting_constraints`：最小统一归一化松弛诊断中冲突的约束，不是不可约冲突集或真实缺口功率。
- `feedback.safety_failures`：求解与AC失败分类；`safety_hard_verified`：本步现场硬约束及AC检查结果。
- `guard_certificate_survived`：执行动作仍满足原始线性区间条件。不得与现场无违例混用。

固定原始候选重放：

```powershell
python -X utf8 scripts/replay_v41.py --trace <原学习策略测试轨迹.jsonl> --checkpoint <对应检查点.pt> --csv data/real/gb/profiles/test.csv --revision 1 --output runs/v41_replay
```

此工具恢复记录中的发令与原始资源权重，不重新调用策略选动作。因此适合检查安全修改影响，不是重新训练后的闭环性能比较。

## 数据与下一步

沿用公开能源/价格/碳曲线驱动IEEE33；没有新增真实EV会话、DR验证或硬件实测数据。小预算DT校准区间可能因极端顶点违反电压边界而无法认证；此时应改进估计、研究条件校准与资源可达性，并在独立数据上重新验证覆盖率，不能调窄区间来追求认证比例。

下一阶段先扩大独立DT拟合与校准样本，完成固定C3条件的控制器比较，再开展长预算多种子和联合C3实验。保持未知偏好与最终测试时段独立。
