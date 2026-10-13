# v4.5：公共库存约束与论文证据流水线

研究问题：在通信和计算受限条件下，DT–C3–Pareto是否能够在明确安全与服务条件下，提供可切换且有竞争力的经济—碳—备用权衡。新版本不保证假设成立，保留强基线、失败与负结果。

## P0：公共终端库存

新配置`configs/v45_ieee33.json`使用`terminal_soc_mode=shared_band`，目标采用dispatch的target_soc，容差默认0.01。当前目标0.55，因此所有偏好、普通MAPPO、固定权重、OLS、集中PPO、MPC、Oracle和备用候选共同受日末SOC区间[0.54,0.56]约束。

共享MILP在完整剩余时域末端施加区间，MPC的费用窗口不改变末端时刻。区间安全层增加储能功率可达边界；这只是必要可达性条件，不证明未来网络或未来扰动可行。现场安全执行器仍验收设备及AC约束。紧急后备可降级终端服务，必须记录短缺并排除出可行前沿；没有硬安全动作时拒绝推进。容差内最多仍允许5kWh库存差，不得声称严格零库存差。

新协议不再计旧经济终端罚项。原始物理向量仍是[-成本/100,-碳kg/100,备用kWh/100]；目标版本增加`-terminal-band-v45`。旧配置保留legacy_penalty，旧结果不得与新问题直接计算提升百分比。下行中断时本地安全器恢复库存属于后备贡献，不是策略表现。

## P1：偏好学习根因

沿用独立偏好编码、向量GAE、PopArt、无效动作掩码、critic修复和训练奖励外生中心化。新增各纯偏好采样步数、梯度范数、梯度余弦和标准化优势均值诊断。余弦为负仅说明当前批次方向相反，不自动证明有害干扰；有一方pure_steps为0时余弦占位0，不得解释为正交。诊断不修改优化梯度。

```bash
python scripts/run_v44_diagnosis.py --config configs/v45_ieee33.json --episodes 32 --seed 43 --eval-days 2 --workers 1 --output runs/v45_diagnosis32
python scripts/report_v44_diagnosis.py runs/v45_diagnosis32 runs/v45_diagnosis32/report.json
```

四案例每个训练预算相同，但三端点家族合计3倍预算；这是根因诊断，不是等家族预算前沿竞赛。

## P2：机制与失效

```bash
python scripts/run_v45_mechanisms.py --group safety --days 2 --workers 1 --output runs/v45_safety
python scripts/run_v45_mechanisms.py --group dt --days 2 --workers 1 --output runs/v45_dt
python scripts/run_v45_mechanisms.py --group c3 --checkpoint runs/v45_diagnosis32/conditioned/latest.pt --days 2 --workers 1 --output runs/v45_c3
python scripts/run_v45_mechanisms.py --group coordinator --checkpoint runs/v45_diagnosis32/conditioned/latest.pt --output runs/v45_coordinator
```

支持安全失败、DT方法与漂移、四种C3模式×带宽/CPU预算、任务/风险协调、IEEE33/69。每个案例保存源码摘要、配置、输入日期、事务账本、完整逐步反馈及结果。仅开放2019-02-27/28开发日期，不读取确认集。

无DT组使用遥测保持、不加残差/区间保护；其零宽度覆盖率不是校准区间覆盖。C3组对冻结策略屏蔽资源动作，属于部署反事实，不代替各消融独立重训。IEEE69组复用区间，明确属于分布转移验收，不能宣称跨网络覆盖保证。计时来自当前主机并受并发影响，能耗仍为模型核算。

## P3：切换与公平对照

```bash
python scripts/run_v45_switch.py --checkpoint runs/v45_diagnosis32/conditioned/latest.pt --csv runs/v45_diagnosis32/development.csv --days 2 --output runs/v45_switch
python run_v45.py --data-root data/real --seeds 1,2,3,4,5 --episodes 120 --eval-days 29 --selection-days 2 --workers 1 --output runs/v45_full
```

切换保留SOC、队列和DT状态，以同一请求任务比较切换、固定初始偏好、固定均衡偏好。它检验连续控制，不证明动态偏好最优性。完整入口保留七种方法；fixed与OLS子策略合计训练预算，选择调用另列，不能声称开发总交互相同。HV参考点冻结，IGD使用验证经验参考集而不冒充真实前沿；服务失败不能进入可行前沿。

## P4：数据、扩展与交付

```bash
python scripts/fetch_v45_ev_sample.py
python scripts/run_v45_external_sample.py --output runs/v45_external_sample
python scripts/audit_v45_data.py --output runs/v45_data_audit.json
python scripts/run_v45_mechanisms.py --group network --output runs/v45_network
```

ORNL发现样本只有26条；当前小时级因果导入保留1条，排除原因完整保留。该样本不是完整站点数据，保留原始来源和SHA。EV、英国能源曲线、SCE响应的组合验收明确标注跨地域/年份，不冒充同站点实测。完整真实EV/DR和目标设备功耗仍为外部证据缺口。

## 一键计划或运行

```bash
python scripts/run_v45_pipeline.py --seeds 1,2,3 --diagnosis-episodes 32 --episodes 120 --eval-days 29 --workers 1 --output runs/v45_pipeline
python scripts/run_v45_pipeline.py --stages P0 P1 P2 P3 P4 --seeds 1,2,3 --diagnosis-episodes 32 --episodes 120 --eval-days 29 --workers 1 --output runs/v45_pipeline --execute
```

第一条只生成具体命令计划；第二条执行。P1与P3是不同预算语义，不能合并作公平比较。外部数据与硬件阻塞不会自动消失，流水线完成也不意味着论文所有结论得到验证。Windows建议Python3.11、UTF-8；`--workers 1`便于降低内存与并发计时干扰。

## 协调器连接修复候选

原服务投影在服务下限可行时直接覆盖风险回退分配，导致该分支的任务／风险优先级不影响实际动作。新增`coordinator_priority_slack_fraction`，默认0保留旧行为；`configs/v45_task_candidate.json`设为0.25，仅让任务风险优先级分配剩余额度的25%，其余75%仍由actor决定，服务下限不变。通过单元测试确认任务能改变分配、actor仍有控制权。是否降低成本需独立验证，不自动改成默认训练配置。

```bash
python scripts/run_v45_mechanisms.py --group coordinator --priority-slack 0.25 --output runs/v45_coordinator_repaired
```

完整EV数据可从 https://ev.caltech.edu/dataset.html 下载JSON，或在本地设置`ACN_API_TOKEN`后：

```bash
python scripts/fetch_acn_sessions.py --site caltech --start 2019-01-01 --end 2019-04-01 --output data/real/ev/acn_complete.json
python -m vpp_research.ev_import --input data/real/ev/acn_complete.json --output data/real/ev/acn_sessions.json --max-kw 7 --dt-hours 1 --replay-actual-departures
```

下载器保留所有分页与SHA，不记录令牌；本轮未取得令牌，未声称已下载全量。7kW是显式设备假设，需按实验站点资料修正。导入仍会排除跨日或缺乏因果申报依据的记录，完整跨日连续调度仍需进一步实现，不得称已全覆盖EV会话。
