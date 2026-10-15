# 文献定位与可检验主张（2026-10-11）

下面是已核查的一组相关一手来源，用于约束贡献表述，不是穷尽性系统综述。不能仅凭这些检索声称“首次”。

| 已有工作 | 可核实范围 | 对本项目的约束 |
|---|---|---|
| Yu 等，The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games，https://arxiv.org/abs/2103.01955 | MAPPO算法基础 | 普通MAPPO保留为对照；不能把官方实现本身列为新增贡献 |
| Felten 等，MOMAland，https://arxiv.org/abs/2407.16312 | 多目标多智能体标准环境与基线 | 多目标/多智能体组合本身不是创新；前沿与偏好泛化需公平验证 |
| Mossalam 等，Multi-Objective Deep Reinforcement Learning，https://arxiv.org/abs/1610.02707 | 深度乐观线性支持、多目标策略集合 | 保留OLS类对照；近似RL子问题不能冒充精确支持集 |
| Tripathi 等，Computation and Communication Co-Design for Real-Time Monitoring and Control in Multi-Agent Systems，https://arxiv.org/abs/2108.03122 | 以AoI连接计算和通信调度 | C3协同不是新概念；需证明其与VPP物理约束、DT区间和三目标的具体耦合价值 |
| A Joint Communication and Computation Framework for Digital Twin over Wireless Networks，https://arxiv.org/abs/2402.00381 | DT相关通信计算联合资源问题 | DT加通信计算的组合不能单独作为首创主张 |
| Wabersich 与 Zeilinger，A predictive safety filter for learning-based control of constrained nonlinear dynamical systems，https://arxiv.org/abs/1812.05506 | 基于MPC与不确定模型的预测安全过滤 | 独立安全层不是本项目首创；现有点态AC检查不能继承该文的理论安全保证 |
| Hua 等，A Digital Twin-based Multi-Agent Reinforcement Learning Framework for Vehicle-to-Grid Coordination，https://eprints.whiterose.ac.uk/id/eprint/234707/1/ICA3PP25_Camera_Ready.pdf | 协作DT的全局预测模型辅助MARL critic；V2G仿真验证 | DT+MARL在能源协调已有直接先例。本项目应聚焦受限C3—校准不确定性—安全执行的因果耦合，不声称隐私保护或纯局部部署 |

## 候选贡献与反证条件

C1：公开遥测驱动的物理残差DT及不确定区间在受限资源下提高控制信息质量。需要估计、覆盖和控制收益同时报告；区间变窄但漏覆盖不能算成功。

C2：在相同通信计算容量下，任务/风险资源分配改变信息质量并带来有价值的经济—碳—备用变化。若充分资源条件下结果相同，这是机制边界；若受限条件下更差，应报告退化而不是删基线。

C3：偏好条件策略在统一安全执行下实现未见偏好的竞争性权衡及在线切换。此项属于第三阶段，本轮不作成立判断。

C4：将候选策略、单步线性区间证书、现场AC验收和可解释拒绝/降级分开。此为具体系统设计与审计贡献，理论只覆盖文档声明的条件性命题。

当前不能声称：新型通用MAPPO；完整多时域安全定理；真实EV/DR部署；真实边缘硬件能效；严格局部观察的分布式执行；已超过所有现有VPP方法。投稿前仍需针对最终方法完成更全面的近期文献对照。
