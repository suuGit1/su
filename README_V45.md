# DT–C3–Pareto VPP v4.5.0

**Digital-Twin-Driven Communication–Computation–Control Co-Design for Safe Pareto Multi-Agent Energy Management in Cyber-Physical Virtual Power Plants**

这是完整的研究仿真运行包：IEEE33＋任务/风险协调器＋能源/通信/计算联合动作＋混合DT＋安全执行器＋普通MAPPO/Pareto-MAPPO。源码、官方MAPPO必要模块、配置、已取得的真实数据和示例检查点均已包含。IEEE69用于扩展验证。

## 1. 安装（Windows PowerShell）

建议Python 3.11。在解压后的 `dt-c3-pareto-vpp-v4.5.0` 根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-v45.txt
.\.venv\Scripts\python.exe scripts/check_v45.py --verify-bundle
```

以下命令中的`python`均指这个虚拟环境的解释器；无需激活环境也可直接替换为`.\.venv\Scripts\python.exe`。首次安装依赖需要网络。包内含真实数据，默认训练不再联网下载、不自动生成合成数据。不要通过上游平台的`vpp`命令或`pip install -e .`替代研究入口。

本次开发验收实际使用Linux CPU、Python 3.12、PyTorch 2.5.1+cpu；Windows路径和UTF-8入口已处理，但没有声称在你的笔记本上实测。CUDA不是必需条件。MILP及AC潮流主要使用CPU。

## 2. 先做小预算验收

```powershell
python run_v45.py --data-root data/real --seeds 43 --episodes 4 --eval-days 1 --selection-days 1 --dt-train-days 2 --dt-calibration-days 9 --workers 1 --output runs/v45_acceptance
```

默认包含七类方法：`mpc milp_oracle ordinary central fixed ols pareto`。普通MAPPO保留官方更新器；fixed和OLS的所有子策略合计占用同一个家族训练预算。Oracle使用真实未来信息，是非因果参考，不能作为同信息MPC解释。4回合只检验流程，不能证明收敛。

首次仅检查参数和数据、不训练，可以追加`--dry-run`并使用新输出目录。查看全部开放参数：

```powershell
python run_v45.py --help
```

## 3. 扩大正式训练

```powershell
python run_v45.py --data-root data/real --seeds 1,2,3,4,5 --episodes 120 --eval-days 29 --selection-days 2 --workers 1 --output runs/v45_full_120
```

`episodes`是每个算法家族、每个种子的总训练回合；`eval-days`是不重复测试日期数，不是循环次数。还可配置学习率、PPO更新次数、rollout回合数、CPU线程、求解限时、测试偏好等，以`--help`为准。训练和选模调用分别记账；不要把“训练预算相同”写成“开发总调用相同”。

同一源码、配置、数据和预算的正常断点可追加`--resume`。配置/版本改变时使用新目录；未知环境调用存在时拒绝自动续训，不能删除失败账本伪造完成状态。

## 4. P0—P4完整实验流水线

```powershell
python scripts/run_v45_pipeline.py --ledger-backend journal --seeds 1,2,3 --diagnosis-episodes 32 --episodes 120 --eval-days 29 --workers 1 --output runs/v45_pipeline --execute
```

去掉`--execute`只生成计划。支持`--stages P0 P1 P2 P3 P4`选阶段、`--config`选配置、`--data-root`选数据目录。`journal`用于会后台同步的工作区；普通本地磁盘也可用默认`sqlite`。事件目录须与检查点一起保存，每个账本只允许一个写入者。

- P0：共享日末SOC区间、安全失败后备、约束和记账检查。
- P1：条件策略与经济/碳/备用端点诊断，逐批次梯度与价值估计记录。
- P2：DT、C3预算网格、任务/风险协调器机制验证。
- P3：连续偏好切换及七基线公平训练、HV/经验IGD报告。
- P4：IEEE69扩展、公开EV发现样本＋DR参数组合验收、数据审计。

流水线跑完并不意味着论文假设全部成立。各阶段的严格范围见`docs/v45/README.md`和`docs/v45/PROTOCOL.md`。

## 5. 无需重训的检查点示例

代码包附有32回合、种子43训练的Pareto检查点及独立开发日期：

```powershell
python scripts/run_v45_switch.py --checkpoint examples/v45/pareto_latest.pt --csv examples/v45/development.csv --days 2 --output runs/v45_example_switch
```

这是开发检查点，不是最终论文最优模型。示例元数据保留原训练信息；代码包不会用该模型替代你的新训练。

## 6. 数据与输出

所有输入位于`data/real/`：GB OPSD能源/价格曲线、NESO碳数据、SCE真实DR参数、公开ACN发现样本及导入审计。国家曲线经过VPP尺度转换，并非IEEE33实测。默认EV和DR关闭；完整因果EV会话不足时不伪造数据补齐。外部组合样本的能源、EV、DR跨地区/年份，不能冒充同站点验证。

`data/real/confirmation_2019Q2/`是冻结确认集，只随包完整保存，v4.5开发入口不使用它。未经冻结最终协议不得用它调参。

主训练输出：`results.json`、`manifest.json`、各方法检查点/轨迹/预算审计、`report.json`、`summary.csv`、`figures/`。流水线还输出`report/REPORT.md`、`report/report.json`、`report/mechanisms.csv`和条件诊断图。重新汇总流水线：

```powershell
python scripts/report_v45.py runs/v45_pipeline
```

已有开发验收摘要位于`release_evidence/v45/`。源码包不含数百MB历史训练中间状态；原始证据另外保留在研究分支，不能仅凭摘要做论文统计。

## 7. 已完成与仍需验证

已完成四组32回合诊断（每组768训练步）、16个C3案例（384步）、6个切换案例（144步）、七基线4回合验收，以及DT/安全/协调器/IEEE69验证。所有失败与负结果保留；资源不足案例按设计拒绝执行。

仍需：充分预算多种子收敛、各C3消融独立重训、稳定的DT控制收益、完整真实EV提前离站/DR实验、目标硬件能耗测量、IEEE69重新校准。现有证据不支持宣称Pareto-MAPPO全面优于强基线或已经满足论文全部结论。

新版本使用共同日末SOC区间[0.54,0.56]，取消旧经济终端罚项。不得与旧目标版本直接混算改进百分比。AC实现安全与区间鲁棒认证分别报告。

MIT许可与第三方归属见`LICENSE`、`vendor/mappo/LICENSE`、`vendor/mappo/UPSTREAM.md`；数据来源与限制见各数据清单。不要把仿真参数称为硬件实测。
