# 第一、二阶段复现入口

使用 research/phase12-v1 分支，安装项目依赖并准备既有 data/real 数据（2019 年 1—3 月完整日）。本轮 CPU 运行，能源控制器为同信息 MPC，不是双 MAPPO 训练。

```bash
python scripts/seal_phase12_data.py
python scripts/run_phase12.py --output runs/phase12_v1 --workers 4
python scripts/report_phase12.py runs/phase12_v1
python scripts/plot_phase12.py runs/phase12_v1
python scripts/test_phase12_contract.py
```

`--workers` 只影响并行度。恢复运行直接使用同一命令；已有案例保留，冻结实现或配置不一致则拒绝续跑。中断不意味着案例失败，实际故障由案例 JSON 的 failed 字段记录。

## 只核查已完成证据

从项目根目录解压 experiments/phase12/mechanism_evidence.zip，然后执行报告、绘图和契约检查命令。证据包含全部设计、实现指纹、离线 DT 结果、逐步轨迹、失败记录、汇总及图表。封存确认集独立保存在 sealed_confirmation_data.zip；本阶段不能对其做性能评估。

summary.json 的 paired_cost 按网络种子先聚合到日期；paired_task_objectives 同时保留成本、碳、备用变化。日期是描述性分析单位，532 个请求并非 532 个独立日期。资源容量及能耗系数属于仿真假设，不是实测硬件值。

原始数据缩放不重新拟合，确认集不插值补缺。协议见 PROTOCOL.md，安全结论边界见 SAFETY_SCOPE.md，逐项验收见 GATE_REVIEW.md。
