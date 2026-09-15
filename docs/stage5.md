# 会话 EV、弃电与 DR：第五阶段

启用 `resource_model: sessions_v1` 和 `network_model: ieee33`。普通 `mappo`、`ippo` 与 `weighted_mappo` 均保留；旧版三智能体环境和旧检查点继续使用 `legacy`。新模型为 ESS、EV、负荷转移、负荷削减、弃光、弃风六个智能体，不能直接加载旧版网络权重。

## 已实现约束

- EV：每车 `[arrival_step, departure_step)` 内充电，单车功率、站级总功率受限，离站前精确交付申报电量。需求单位为充电桩侧 kWh，不推断电池 SOC，也未实现 V2G。接入后需求与申报期限视为已知且不变；跨日会话须显式分配当日电量，不能直接截断。
- DR 转移：正功率延后用电，负功率补回；欠供电量不能为负，有容量和累计延后电量上限，日末必须补回。延后与补回互斥。默认效率 1；未包含温控舒适度、用户响应概率和逐任务最长等待时间。
- DR 削减：永久削减单列功率、日能量预算、补偿费用。削减与延后合计不得超过原始负荷的一定比例。
- 弃光、弃风分别非负且不超过当步可用出力；接入各自 IEEE33 节点，进入电量平衡、弃电费用和 AC 潮流审计。
- 电网购电功率 = 原始负荷 − 光伏 − 风电 − ESS 放电 + EV 充电 − DR 转移 − DR 削减 + 弃光 + 弃风。

`FlexSpec` 是新增资源参数入口；ESS、网络和价格参数继续采用已有 `DispatchSpec`。新模式不再使用旧聚合 EV 电池及旧 DR 费率。

## MILP/MPC 与安全层

三者共享六资源网络矩阵和 MILP 约束。日前 MILP 拥有整日实际曲线和全部会话，是完美预知参考。MPC 只读取当前曲线与已接入会话；经济窗口为 `lookahead`，另用当前值持久性预测延伸到日末，确保已知 EV 期限和 DR 补回在预测下可行。尾部没有运行费用目标；窗口末端 ESS 目标仍是启发式。

策略动作经同一 MILP 做归一化 L1 投影，规划已知任务的剩余时段。无法找到可执行解时明确报错；未实现自动拒绝车辆、紧急负荷切除或超时后备控制。未来突发接入、预测误差可能使下步不可行，因此不声称递归鲁棒可行。外部物理动作字典用于回放求解器分车计划；不应作为未经验证的任意安全控制接口。

三目标中的备用在新模型下采用下一步可激活功率，尾部保留 EV/DR 约束；`reserve_hours` 必须等于 `dt_hours`。该定义有独立版本标识，日末备用记零。线性可行与 AC 安全分别记录。

## 直接运行

先按仓库原有说明安装 MAPPO 和 IEEE33 依赖，再执行：

```bash
python -m vpp_mappo train --config configs/sessions_smoke.json --output runs/sessions_mappo
python -m vpp_mappo.compare --checkpoint runs/sessions_mappo/latest.pt --output runs/sessions_compare --episodes 1 --lookahead 2
```

比较目录包含普通 MAPPO、MPC、MILP 的独立轨迹和配对成本；轨迹及回合汇总记录离站欠充、交付电量、DR 延后/补回/削减电量、日末欠供、弃电和安全层求解耗时。冒烟训练不代表收敛或论文性能。

## 真实会话接入

真实日曲线模式必须提供配套会话文件。规范化 CSV 字段为：

```csv
scenario,id,arrival_step,departure_step,energy_kwh,max_kw
2024-01-01,vehicle_1,8,17,20,7
```

日期需与电力 CSV 的场景键精确相同；步长、时区和电量口径须统一。departure_step 是接入时申报的期限，energy_kwh 是当时声明的需求；不要将事后离站时间和已充电量自动替代为在线已知字段。如果原数据缺少申报字段，应明确标记代理需求实验，并记录生成规则。

```bash
python -m vpp_mappo.prepare_ev --csv sessions.csv --output sessions.json --horizon 24 --dt-hours 1 --source '数据出处；接入时申报字段；时间离散化规则'
```

训练配置填写 `ev_sessions_path`。JSON 必须声明 `schema_version: 1`、`energy_basis: grid_kwh`、`source`、`horizon`、`dt_hours` 和 `scenarios` 字典。零车辆场景显式填 `[]`。全部记录验证单车可充性，集体/网络可行性在求解时验证。

评估/比较可用 `--csv test.csv --ev-sessions test_sessions.json` 指定独立测试数据。检查点保存资源参数和会话快照，不依赖原参数文件持续存在。默认无数据模式生成两个明确标记的示例会话，未接入真实 EV 数据。场景文件虽包含未来会话，在线观测与求解入口只筛选已接入车辆。当前 EV 观测为需求、数量和最近期限的聚合，属于部分可观测控制，不包含完整单车状态编码。
