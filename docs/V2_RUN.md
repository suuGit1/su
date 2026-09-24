# v2完整运行版

项目：Digital-Twin-Driven Communication–Computation–Control Co-Design for Safe Pareto Multi-Agent Energy Management in Cyber-Physical Virtual Power Plants。

v2保留IEEE33、规则式任务/风险协调器、18个C3角色、物理与残差DT、区间投影/现场MILP/AC执行器，以及独立的普通MAPPO和Pareto-MAPPO。统一入口增加按种子并行、共享DT、断点续跑和合并报告。

## 安装与启动

从项目根目录运行，使用Python 3.11。当前研究训练入口验证的是CPU；历史CUDA11.3代码与依赖不要混入这个环境。

```bash
python -m venv .venv
# Linux/macOS 激活环境；Windows请使用 .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements-research-lock.txt
python scripts/prepare_real_data.py --verify-only
python run_v2.py --seeds 1,2,3 --episodes 12 --eval-days 3 --selection-days 1 --workers 3 --dt-train-days 31 --dt-calibration-days 12 --output runs/v2
```

`data/real`目录必须存在。原始数据大文件通过之前的数据包提供，已校验的训练/验证/测试CSV在仓库中；数据校验失败会明确报错，不会回退合成曲线。

扩大预算示例（没有预先声称这组100回合实验已完成）：

```bash
python run_v2.py --seeds 1,2,3,4,5 --episodes 100 --eval-days 29 --selection-days 2 --workers 3 --output runs/v2_100
```

仅运行双MAPPO或检查参数：

```bash
python run_v2.py --methods ordinary pareto --seeds 1,2,3 --episodes 12 --eval-days 3 --output runs/v2_dual --dry-run
```

去掉 `--dry-run` 开始执行；若已有预检/运行目录，请加 `--resume`。同目录续跑必须保持实验协议与预算不变。更换预算、数据或配置使用新目录。`--workers`按种子并行，`--cpu-threads`控制每个训练进程的线程数；本机资源不足时设 `--workers 1`。

其他开放参数：`--methods`、`--learning-rate`、`--ppo-epochs`、`--solver-time-limit`、`--eval-preferences`、`--ols-policies`、`--hv-reference`、`--ev-sessions`、`--dr-mode`。偏好示例：`--eval-preferences "0.2,0.3,0.5;0.6,0.1,0.3;0.1,0.7,0.2"`。

## 本版关键修正

1. 物理DT按DR设备功率及累计预算限制候选指令外推；关闭DR时不会产生虚假的累计移峰/削减量。DT校准尺度只从训练残差RMSE估计，仍使用独立日期块校准。
2. SOC不确定区间与已知物理支撑集求交，不要求安全层覆盖不可能存在的SOC。不会为追求可行率而裁剪未知负荷或风光误差。
3. PCC备用全部使用**该执行时段起点状态、该时段曲线和该时段实际AC基线**。实际执行计划可作为零激活的已验证见证；其他激活比例仍须MILP和AC核验。最后一个时段也按相同口径计算。
4. 备用表示同一时段起点可选择的替代调度能力，持续一个调度步；不是执行结束后额外获得的备用。有限激活点检查不等于连续区间或全扰动安全证明。
5. 目标升级为 `c3-ac-pcc-same-period-sampled-reserve-v4`，DT升级为 `ridge-block-conformal-physical-dt-v2`。旧模型拒绝混用，必须重训。

## 七种方法与预算

默认运行 `mpc milp_oracle ordinary central fixed ols pareto`。普通MAPPO、集中式PPO、MPC使用经济目标；固定权重与OLS使用MAPPO子策略；Pareto-MAPPO使用偏好条件actor及向量critic/GAE。完美预知MILP明确为非因果参考。

`--episodes`是每种方法、每个种子的总训练回合。固定权重三个子策略和OLS全部子策略共享此预算。OLS若提前没有新角点或验证失败，会报告未用完预算，不补造训练。策略选择交互单列，因此这里统一的是训练预算，不是开发总交互。确定性优化基线在不同训练种子下的重复结果不构成独立训练证据。

DT训练取训练集前段；校准取验证集前段；后续独立日期用于策略选择；测试使用独立月份。HV参考点预先固定，IGD参考集来自策略选择日期的经验前沿。物理或服务约束失败、备用未确认的偏好整组不进入有效前沿；空前沿HV=0、IGD=null。

## 输出、验证与模型

根目录输出 `manifest.json`、`release.json`、`results.json`、`report.json`、`summary.csv`、`report.md`；各 `seed_N` 保存方法结果、训练日志、轨迹、`latest.pt`及回合边界 `resume.pt`。运行异常的方法可在 `--resume` 时重试，旧记录与中断控制器轨迹保留；无合格验证策略的结果按原协议保留。仅有 `latest.pt` 可用于推理；训练续跑还需 `resume.pt`和相同协议。当前续跑校验包含原始绝对路径，跨机器搬迁模型后建议先用下面的推理命令；跨目录训练续跑尚未自动迁移。

```bash
python -m vpp_research.validate_v2 runs/v2 --days 3
python -m vpp_research.real_report runs/v2
python -m vpp_research.plot_v2 runs/v2
python -m unittest discover -s validation -p "test_*.py"
```

本轮独立3日DT/MPC对照中：归一化估计RMSE从0.02213降至0.02104；相同MPC的三日AC成本合计下降约4.3%；最终动作区间证书从51/72步增至64/72步，仍有8步不能认证。三日区间覆盖均达到100%，样本少且存在策略/分布变化，不能外推为普遍覆盖保证。完整回归118项通过。

真实EV完整会话仍缺失，默认关闭；同地区DR参数未齐，SCE只能显式用作跨地区敏感性来源。能源曲线为英国国家曲线缩放到VPP，不是IEEE33现场数据。下行/CPU/能耗参数仍为仿真假设。v2是可运行研究框架与扩大验证版本，不是全部论文实验已完成。

模型包解压到项目目录后，可在当前真实数据上复现单日闭环：

```bash
python run_vpp.py run --config configs/v2_ieee33.json --method ordinary --checkpoint v2_models/seed_1/ordinary_1/latest.pt --day-index 0 --output runs/v2_replay_ordinary
python run_vpp.py run --config configs/v2_ieee33.json --method pareto --checkpoint v2_models/seed_1/pareto_1/latest.pt --day-index 0 --output runs/v2_replay_pareto
```
