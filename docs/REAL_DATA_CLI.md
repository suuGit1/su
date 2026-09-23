# 真实数据与开放命令行

默认真实数据集中在 `data/real/`，无需手工拼接先前实验目录。新增多种子入口：

```bash
python scripts/run_all_seeds.py --seeds 1,2,3 --episodes 100 --eval-days 29 --output runs/real_100
```

默认运行观察等价 MPC、普通 MAPPO 和 Pareto-MAPPO。先训练与校准一个共享 DT，再逐算法/种子训练并在独立测试日期运行。训练曲线不足100天时按顺序循环31个训练日；不读取验证/测试日训练策略。

建议先检查预算，不启动训练：

```bash
python scripts/run_all_seeds.py --seeds 1,2,3 --episodes 100 --eval-days 29 --dry-run --output runs/preview
```

正式运行使用新目录；已有相同协议目录须显式 `--resume`。当前多种子恢复要求预算协议不变；修改总预算请使用新目录，或用单模型 `run_vpp.py train --resume --episodes ...` 扩展训练。失败的方法结果会保留，不自动删除重跑。半回合/测试日恢复不是无缝的，失败目录应保留后重新启动新实验。

|参数|含义/默认值|
|---|---|
|`--config`|基础模型、协调器和C3配置；默认 `configs/integrated_ieee33.json`|
|`--data-root`|统一数据根目录；默认 `data/real`|
|`--methods`|空格分隔算法，可选 `mpc ordinary pareto`|
|`--seeds`|逗号分隔训练种子，例如 `1,2,3`，不限定数量|
|`--episodes`|每算法每种子的训练总回合数；一个真实日24步，默认100|
|`--eval-days`|不重复测试日数，默认29；不得超过数据中有效日期数|
|`--eval-preferences`|分号分隔三目标权重，如 `"0.2,0.3,0.5;0.6,0.1,0.3"`|
|`--dt-train-days`|用于离线DT拟合的训练日数量；默认全部31天|
|`--dt-calibration-days`|独立校准日数量；默认27天，90%场景块校准至少9天|
|`--learning-rate`、`--ppo-epochs`|覆盖学习率与每次更新的PPO迭代数|
|`--cpu-threads`|CPU线程数；当前研究训练入口仍使用CPU|
|`--solver-time-limit`|单次优化器时间上限（秒），不是端到端硬实时保证|
|`--ev-sessions`|真实会话JSON，必须覆盖全部日期；不提供时明确关闭EV|
|`--dr-mode`|`off` 或 `sce-derated`，默认off；后者为显式跨地区历史响应降额|
|`--resume`|恢复相同实验协议，复用已完成单元和回合边界检查点|
|`--dry-run`|只核对数据、拆分和预算，不训练|
|`--output`|完整实验输出目录|

例如更改算法、学习率与偏好：

```bash
python scripts/run_all_seeds.py --methods ordinary pareto --seeds 1,7,19 --episodes 200 --eval-days 20 --learning-rate 0.0001 --ppo-epochs 5 --eval-preferences "0.2,0.3,0.5;0.6,0.1,0.3" --output runs/real_custom
```

独立测试的普通 MAPPO 仍使用经济策略；遍历偏好只是相同测试请求下的对照，不会将普通MAPPO改造成偏好条件策略。MPC为固定经济优化，不按偏好重复执行。默认三种测试偏好下，两种RL方法每个种子各执行 `29×24×3` 测试交互，MPC执行 `29×24`；训练步数、测试开销分开报告。

## 原集成入口也改为默认真实数据

```bash
python run_vpp.py prepare --output runs/real_dt
python run_vpp.py train --method pareto --dt-model runs/real_dt/residual.json --episodes 100 --output runs/pareto_real
python run_vpp.py run --method pareto --checkpoint runs/pareto_real/latest.pt --day-index 0 --output runs/replay_real
python run_vpp.py run --method mpc --dt-model runs/real_dt/residual.json --day-index 0 --output runs/mpc_real
```

`--day-index` 从0开始指向测试CSV按日期排序后的指定日。真实数据缺失会报错；合成演示须显式加 `--synthetic`，例如 `python run_vpp.py all --synthetic --output runs/synthetic_demo`。旧合成DT模型不能直接用于新的真实入口；模型中保存并校验真实数据/EV/DR协议指纹。

输出包括数据与配置协议、DT原始训练标签、双算法模型、训练轨迹、每个算法/种子的测试结果与汇总。`all_successful` 只表示执行及物理/服务检查通过；`all_objectives_confirmed` 还要求备用确认全部通过。备用未确认步数单独保留，不能把其混成完整可行Pareto前沿。

本次实际功能验证仅运行种子71、每算法1回合训练、1个独立测试日、1组测试偏好，DT使用2个拟合日及9个校准日。它验证了数据与程序连接，不是正式100回合多种子结果。真实日曲线下已出现部分备用无法确认的步骤；需要后续分析，不能声称全部三目标指标已通过。

## 本次校验记录

110项回归测试通过。OPSD完整原始文件重新下载并通过固定SHA256校验（130339665字节）；从原始OPSD、NESO日文件重建后，训练/校准/测试三个CSV哈希与交付文件完全一致。

真实日闭环无交流/资源约束越限；普通MAPPO有2步、Pareto有5步备用未确认，MPC有1步。保留这些标记，不能将该日当作全部三目标均确认的Pareto证据。
