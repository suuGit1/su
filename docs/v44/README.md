# v4.4：先修学习机制，再扩大预算

本版本继续保留 IEEE33、DT、公开信息协调器、C3时序、安全执行层、普通MAPPO、固定权重MAPPO、OLS、集中PPO、MPC与Oracle入口。修复针对Pareto学习根因，不删除更强对照，不更改经济—碳—备用的单位、符号或终端罚项。

## 实现

- `learning_cleanup`：保持80维观察，停用通道的AoI/队列置零；按已知资源配置屏蔽永久无效动作。能源候选置零与策略损失掩码一致。官方MAPPO使用上游active_masks；集中PPO屏蔽对应联合概率和熵；新Pareto更新器同时屏蔽PPO、熵、KL与裁剪统计。偶发安全修正不等于动作无效，不做事后掩码。
- `conditioned_v44`：偏好独立编码后融合，避免与AoI共同做输入LayerNorm。AoI/队列使用固定asinh变换，不拟合测试数据。保留向量critic/GAE与原单位标量化。
- 偏好采样：前四回合明确覆盖三个顶点与均衡点；随后覆盖三个近顶点及一个随机内部偏好。
- 同学习器固定端点：`pareto_fixed_preference`固定训练及推理条件，保持网络、初始化、更新规则相同；独立于官方固定权重基线。
- critic修复：输出层初始化增益单独配置；首次价值拟合共200次，后续每批在5次常规更新外增加25次。actor仍使用采样时的GAE，不利用更新后的critic回写旧优势。额外价值拟合不增加环境交互，但增加计算，报告中必须保留更新次数与训练耗时。
- 新记录：有效动作数、偏好动作跨度、三目标优势标准差、价值解释方差、实际actor/critic更新次数；开发报告拆开运行成本、终端罚项、终端SOC、碳和备用，以及最后两步发出/收到/执行的储能动作。
- 所有新配置默认关闭或保持旧值，旧v4.3检查点仍可加载。v4.4完整配置显式启用上述修复。改变源码或配置后，旧冻结训练目录拒绝直接续训；用新目录或检出原提交复现。

## 运行

建议Python 3.11及仓库原有依赖。先执行机制诊断：

```bash
python scripts/test_pareto_v44.py
python scripts/run_v44_diagnosis.py --episodes 8 --seed 43 --eval-days 1 --workers 4 --output runs/v44_diagnosis
python scripts/report_v44_diagnosis.py runs/v44_diagnosis
```

该诊断使用真实一月训练数据、未用于DT校准的二月开发日及已冻结的残差DT；不读取四月至六月确认性测试集。默认四案例为同一学习器的经济、碳、备用固定端点和条件策略。每案例训练预算相同，但三个端点合计预算为条件策略三倍：这是机制诊断，不是公平总预算下的前沿优越性实验。无选模；全部使用末次检查点。

诊断训练量可改为32或64回合。配置通过`--config`指定。`configs/v44_architecture_only.json`用于关闭额外价值更新的消融。每个案例默认回合边界保存检查点；相同命令可恢复，协议改变或账本存在未知交互会拒绝续跑。未知交互必须保留并在新目录重跑，不得删除后冒充完整预算。

完整系统入口示例：

```bash
python run_v44.py --data-root data/real --seeds 1,2,3,4,5 --episodes 120 --eval-days 29 --selection-days 2 --workers 1 --output runs/v44_full_120
```

完整入口默认保留七种方法；`--episodes`沿用原入口的训练预算语义，选模交互另列，不是训练与选模的合计预算。此命令是可运行入口，不代表已完成该规模实验。当前不建议在机制诊断未通过前直接扩大五种子长预算。critic配置项在JSON中：`critic_initial_gain`、`critic_warmup_epochs`、`critic_extra_epochs`。计算预算不同必须报告，不能仅以环境交互数声称计算成本相同。

## 必须正确解释的现象

偏好响应增大不等于响应方向正确；critic拟合改善不等于控制改善；单日、单种子不能证明收敛。纯碳/纯备用偏好下，经济分量中的终端SOC罚项权重为零，因此低终端SOC可能是目标定义允许的行为。若论文要求跨日能源中性，必须把终端SOC改成所有算法共有的服务约束并重新冻结协议，同步优化对照；本版本没有悄悄做这个改变。

架构修复首轮与价值修复结果分别保存，失败尝试与未知交互也保留。`architecture_source.zip`保存首轮冻结源码，以免后续价值修复使首轮无法复现。

## 进一步定位后的候选配置

外生基准候选已通过单种子八回合实跑：同偏好配对比较总成本下降约1.4%—2.3%、备用提高约30%—32%；碳变化很小且方向并非全部改善。保留为独立配置，不自动替换默认版本：

```bash
python scripts/run_v44_diagnosis.py --config configs/v44_centered_candidate.json --episodes 32 --seed 43 --eval-days 2 --workers 1 --output runs/v44_centered_diagnosis
```

该命令会把中心化同时用于条件策略与三个同学习器端点，才可进一步比较学习质量。完整系统也可通过`run_v44.py --config configs/v44_centered_candidate.json ...`使用同一开关。原始物理目标仍用于评估与选模，中心化critic输出只表示学习回报，不能解释成绝对物理成本预测。

配对日期候选为`configs/v44_paired_candidate.json`，已完成调度逻辑测试及4回合端到端验收（96步训练、96步开发评估），多日期控制收益未验证。其每8回合只覆盖一个外生日期，不能套用普通采样的独立日期数。
