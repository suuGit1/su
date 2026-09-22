# Stage 10：多目标接口与可审计 Pareto 解集

项目：Digital-Twin-Driven Communication–Computation–Control Co-Design for Safe Pareto Multi-Agent Energy Management in Cyber-Physical Virtual Power Plants。

## 本轮完成

- 新增 `vpp_research.parallel_env.VPPParallelEnv`，按 MOMAland 公开 Parallel 接口组织观察、向量奖励、终止标记与逐智能体空间。独立实现于 PettingZoo；未安装或认证上游 MOMAland 集成。
- 每个智能体收到同一团队三维奖励：原始目标减去各维共享的违规惩罚。团队回报只能计一次，不可跨智能体累加。`info.raw_objective_vector` 保留无惩罚目标用于评价。动作是原始高斯动作，内部统一 tanh。
- `state()` 仅返回公开观察拼接，不向 critic 增加物理真值；`infos` 包含事后评价信息，不得作为在线观察。
- 新增 `pareto_archive`：475 个“训练种子×方法×偏好”记录，均具有三场景完整可行评价；其中450个标记未见偏好。保留逐场景失败/可行性指标与数据源 SHA256。未见是相对于有限训练偏好集合，不代表跨分布泛化已成立。
- 生成三目标两两投影：预先固定最小训练种子1用于展示，其他种子全部保留在JSON。圆圈表示各方法在三维中的非支配点；二维投影不能单独证明三维支配关系。固定权重方法代表多个模型的策略集合。
- 普通官方 MAPPO 的训练、配置、检查点及评价路径没有替换。

## MOMAland 借鉴边界

参考其[向量奖励 Parallel API](https://momaland.farama.org/api/parallel/)与[学习算法说明](https://momaland.farama.org/tutorials/learning_algorithms/)，访问日期2026-09-22。MOMAland 列出的 MOMAPPO (OLS) 为多策略、团队线性效用；本项目已有算法为单个偏好条件 actor + 向量 critic/GAE。两者不可混称。OLS 基线尚待实现，且总训练预算必须覆盖它训练的所有子策略。当前固定权重集合不是 OLS。

## 尚未完整呈现的内容及验收顺序

|顺序|缺口|已有基础|下一步验收|
|---|---|---|---|
|1|统一可交付备用口径|v1线性代理；v2 AC端点检查及112次激活无失败样本|含网损PCC功率跟踪、整段轨迹与连续范围风险验证；冻结目标后重训，v1/v2禁止混表|
|2|安全失败闭环|约束投影、AC复核、经济后备|预测误差/资源不足/求解超时下明确服务降级，验证紧急控制；不宣称递归可行已证明|
|3|Pareto基线与收敛|普通MAPPO、固定权重、集中PPO、偏好条件策略；5种子短预算|新增OLS，多子策略总预算匹配；验证集选型、独立测试；检查偏好变化是否产生有意义权衡|
|4|DT控制收益与覆盖|残差估计误差下降、分块校准、7种压力机制|覆盖在策略变化与漂移下达标；配对控制成本置信区间支持收益，不能只给RMSE|
|5|真实EV/DR完整证据|因果EV导入和物理约束；真实GB能源碳曲线|完整合法会话、提前离站事件、真实DR参数；不同地区数据不得伪装成同一站点|
|6|C3工程证据|通信/计算/联合消融、延迟与过期实验|实际设备的推理、通信、求解时间与能耗校准；更大网络的扩展验证|
|7|论文最终统计|统一参考HV/IGD与5种子报告|长预算学习曲线、更多独立日期和置信区间；参考前沿仍须称经验参考|

现有证据不支持“所有任务完成”：九方法族每族480交互步仍短；联合方法相对普通MAPPO的HV配对置信区间跨零。真实GB数据残差DT降低估计误差约4%，控制成本差异尚无显著改善，测试覆盖约75%。本轮只是接口和证据呈现增强，没有新增训练收益结论。

## 运行

在项目依赖已安装的环境、仓库根目录执行：

```python
from vpp_mappo.config import Config
from vpp_research.parallel_env import VPPParallelEnv

env = VPPParallelEnv(Config.load('configs/research_smoke.json'))
observations, infos = env.reset(seed=19)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    observations, vector_rewards, terminated, truncated, infos = env.step(actions)
env.close()
```

```bash
python -m unittest discover -s validation -p 'test_stage*.py'
python -m vpp_research.pareto_archive --runs PATH_TO_STAGE9_RUNS --output experiments/stage10
```

报告输入是此前交付的 Stage9 实验包中的原始 campaign/ablations 结果，不需重训。报告明确限定旧 stage9 协议，图与JSON仍使用v1线性备用代理；不得作为v2的结果。

验证：全阶段回归共86项测试通过（26.729秒）；新增三项覆盖整回合轨迹等价、无效动作与输出隔离、缺失/重复/失败场景的解集筛选。报告精简字段后再次运行新增三项测试通过。上游 MOMAland 训练器直接接入尚未测试。
