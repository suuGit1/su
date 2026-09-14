# 官方 MAPPO 主框架

| 文件 | 职责与扩展位置 |
|---|---|
| vpp_mappo/config.py | JSON 配置与字段检查 |
| vpp_mappo/data.py | CSV 场景验证、读取、哈希 |
| vpp_mappo/environment.py | 局部观测、集中状态、动作缩放、奖励、安全接口 |
| vpp_mappo/algorithms.py | 官方策略、更新器、缓存封装与算法注册 |
| vpp_mappo/runner.py | 完整回合训练、保存、独立评估 |
| vpp_mappo/__main__.py | train / evaluate / doctor 入口 |
| vendor/mappo | 锁定版本的官方实现 |
| src/vpp/dt_c3 | 已修复的原有物理环境与待验证历史模块 |

## 环境替换契约

train 和 evaluate 接受 env_factory(config, csv_path)。环境提供 num_agents、单智能体 observation_space/share_observation_space/action_space，以及 data（无外部数据为 None）。reset(seed, profile_index) 返回 (obs, share_obs)；step(raw_actions) 返回 (obs, share_obs, rewards, done, info)。数组首维为智能体数，奖励为 (N, 1)，集中状态按智能体重复；info 至少含 cost、reward、constraint_violations。CSV 对象提供 sha256、profiles。

本版假定每个任务恰好 horizon 步且全部智能体共同终止，不支持异步退出或可变长度回合；扩展这些行为时需调整缓存和终止掩码。

## 算法与物理约定

采用官方 feed-forward 模式，单个并行环境、共享网络、GAE 和 ValueNorm。每个完整调度任务更新一次。缓存保存原始高斯动作及其 log probability；环境再做 tanh、物理缩放与可选安全修正，避免用修正后动作匹配修正前概率。

三个智能体为 ESS、EV 聚合资源、DR。10 维局部观测包含公共时间/电价/聚合净负荷/AoI/丢包率、自身状态与目标、三维身份编码；不直接读取另一储能资源 SOC。集中 critic 使用局部观测拼接的 30 维状态，不额外读取真实全状态；IPPO 使用 10 维局部 critic。公共广播仍属于执行所需信息。

日末是含终端 SOC 罚项的真实有限时域终点，masks=0，不跨回合 bootstrap。SOC 目标 0.55 是软惩罚，不是 EV 离站硬约束。奖励为经济费用与违规/终端罚项之和取负并缩放。经济费用单独记录，不能将 reward 当电费。默认动作尺度：ESS ±250 kW、EV ±300 kW、DR 0–150 kW；费用系数在适配器中明确列出，后续需配置化并按数据校准。

## CSV 接入

UTF-8 表头及示例行：

```csv
scenario,step,load_kw,pv_kw,wind_kw,price
day_001,0,800,0,100,0.3
day_001,1,820,0,110,0.3
```

示例仅展示格式，默认训练需每个 scenario 恰好 96 行。step 必须为 0 到 horizon-1，不能重复或缺步。功率单位 kW、电价为统一货币/kWh，负电价允许；功率非负且所有值有限。时间间隔必须与 dt_hours 一致，原始时间戳、时区、缺失值和量纲由预处理负责。

复制基线配置并增加 `"train_csv": "data/train.csv"`。评估命令增加 `--csv data/test.csv`。训练可循环使用场景，测试回合数不得超过独立场景数。程序拒绝训练测试同路径/同内容 CSV、合成训练测试种子重叠。哈希无法识别部分重叠或重排后的文件，仍需按完整日期/场景划分 train/validation/test。

默认训练和评估都使用合成数据，输出明确标注 synthetic；真实测试必须显式提供 --csv。本阶段不要求下载外部数据。

## 研究边界与后续顺序

safety、delay_steps、packet_loss、digital_twin 为配置开关；默认开启安全，关闭 DT，无延迟无丢包。通信和计算资源动作固定，原有轻量 DT 未经过训练与论文级验证，所以当前不构成 DT-C3 联合优化。

当前采用聚合功率平衡，未启用线路/电压约束，也没有 IEEE 配电网、EV 到离站、DR 反弹、已核验 MILP/MPC、碳目标/Pareto 前沿、LLM/RAG 代理或异常通信安全保证。检查点虽保存优化器，但仅实现评估加载，尚无精确续训。

下一步先统一物理约束与 MILP/MPC，对接真实日曲线和标准配电网，再做 MAPPO、局部 critic 和安全开关的多种子对照；随后引入可测量贡献的 DT、通信计算联合决策与必要的工具代理。接口可扩展不等于上述研究模块已实现。
