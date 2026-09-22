# Digital-Twin-Driven Communication–Computation–Control Co-Design for Safe Pareto Multi-Agent Energy Management in Cyber-Physical Virtual Power Plants

第十二阶段将 OLS 自适应验证纳入四方法总预算，已运行同预算流程验证。[预算规则与运行命令](docs/stage12.md)。

第十一阶段新增含网损 PCC 三目标、紧急后备、OLS 子策略、提前离站事件和真实 SCE DR 数据分析。[实现与未完成验收](docs/stage11.md)。

第十阶段新增 MOMAland 风格向量 Parallel 接口、偏好解集和前沿投影；普通 MAPPO 保留。[当前缺口与验收顺序](docs/stage10.md)。

第九阶段已补充公开英国能源/碳数据、C3 四组消融、集中式 PPO、备用激活与 AC 确认、下行/CPU 时序、再校准及数值问题修复。**研究验收仍有未通过项，不能宣称全部 P0/P1 已完成。** 见 [第九阶段协议](docs/stage9_protocol.md) 与 [验收报告](docs/stage9.md)。

第八阶段已实现 DT 残差/场景分块校准、固定 65 维观察、偏好条件 actor 与向量 critic/GAE，以及五种子/未见偏好/HV/IGD 和 AC 执行前校核。**估计改善已观察到，控制与 Pareto 性能优势尚未证实。** 见 [第八阶段实测报告](docs/stage8.md) 与 [复现协议](docs/research_protocol.md)。

第七阶段已加入观察等价聚合 MPC、影子决策审计及规则式任务/风险协调器。见 [第七阶段说明与 Pareto 路线](docs/stage7.md)。

路线 A 第一批已接入通信/计算队列、基础物理 DT 和普通 MAPPO 联合决策。见 [第六阶段说明](docs/stage6.md)。项目标题是最终研究目标；第八阶段已补齐 Hybrid DT 和 Pareto 主体，论文级鲁棒安全验证仍待完成。

**第四阶段已保留普通 MAPPO，新增三目标记录、固定权重多策略对照和向量奖励接口。见 [第四阶段说明](docs/stage4.md)。偏好条件 Pareto-MAPPO 已在第八阶段单独新增，进度按验收清单如实记录。**

**第三阶段已接入 IEEE33、统一 MILP/MPC、真实 OPSD 下载与清洗、配对比较和交流潮流审计。完整命令与模型假设见 [第三阶段运行指南](docs/stage3.md)。**

以下保留第二阶段聚合模型的快速入口。第二阶段已提供三资源智能体训练、模型保存、独立评估和 CSV 数据接口。策略网络、PPO 更新、GAE 缓存与 ValueNorm 来自 [官方 MAPPO 固定版本](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac)，保留上游 MIT 许可证。

已在 Python 3.12、PyTorch 2.5.1 CPU 上真实运行。默认使用合成场景，无需下载数据。这是可运行研究基线，尚未完成论文性能验证。

## 安装与直接运行

```bash
git clone --branch research/stage9-c3-public-data https://github.com/suuGit1/su.git
cd su
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`；Linux/macOS 使用 `source .venv/bin/activate`。建议 Python 3.10–3.12，本次实测 3.12。Linux/Windows CPU 安装及运行：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-mappo.txt
python -m vpp_mappo doctor
python validation/verify_vendor.py
python validation/test_stage1.py
python validation/test_stage2.py
python -m vpp_mappo train --config configs/mappo_smoke.json --output output/smoke
python -m vpp_mappo evaluate --checkpoint output/smoke/latest.pt --output output/smoke_eval
```

macOS 跳过第一条 CPU 索引命令，直接安装 requirements。命令均在仓库根目录执行。本入口无需安装旧 Web 服务、求解器或旧 CUDA 11.3 依赖。输出目录非空会报错，请使用新目录。

训练输出：config.json、metadata.json、training.csv、latest.pt。评估输出：episodes.csv、trajectory.csv、summary.json。记录官方版本、依赖版本、完整参数和数据哈希。检查点支持独立评估；暂未实现断点续训。

## 日调度与对照

```bash
python -m vpp_mappo train --config configs/mappo_baseline.json --output output/mappo_s1 --seed 1
python -m vpp_mappo train --config configs/mappo_baseline.json --algorithm ippo --output output/ippo_s1 --seed 1
python -m vpp_mappo evaluate --checkpoint output/mappo_s1/latest.pt --output output/mappo_test --episodes 30 --seed 100000
python -m vpp_mappo evaluate --checkpoint output/ippo_s1/latest.pt --output output/ippo_test --episodes 30 --seed 100000
```

默认基线为 96 步 × 15 分钟、100 回合，是起始预算，不代表收敛。对照应使用相同测试场景，并另外执行多个训练种子。MAPPO 为局部 actor、集中 critic；IPPO 为局部 actor、局部 critic。两者都共享资源间网络参数并使用身份编码，不能把这里的 IPPO 描述成三套独立网络。

兼容 CUDA 的 PyTorch 可使用 `--device cuda`，不可用时明确报错；本次仅验证 CPU。旧实验目录属于历史代码，新统一入口不调用其同名算法。

真实数据接入、扩展接口与研究边界见 [框架说明](docs/stage2.md)，实测记录见 [验证记录](docs/stage2_validation.md)。

本项目基于用户上传的 virtual-power-plant-dt-c3-gpu-cu113.zip；原项目来源为 [vinerya/virtual-power-plant](https://github.com/vinerya/virtual-power-plant)，保留上传包许可证和作者元数据。上传包没有可核验原始提交号。[第一阶段说明](docs/stage1.md) 记录了环境与指标修复。

## 会话 EV、DR 与弃电

第五阶段运行入口及真实会话格式见 [阶段五说明](docs/stage5.md)。普通 MAPPO 保留，新增六资源模型通过 `configs/sessions_smoke.json` 启用。
