# v3交付验收

- 127项回归通过；随后完成快照、载荷校验及双MAPPO恢复专项10项通过（有重叠，不相加）。
- two_hour_acceptance：最终真实日曲线前两个小时的七方法功能验收。不是完整日或长期性能证据。
- partial_24h：整日运行只完成五种方法，OLS含中断未知调用，整组未通过。
- pre_payload_failure：追加日志缺项的负例，未通过。保留历史问题，不补造记录。
- diagnostics_ordinary / diagnostics_pareto：冻结v2种子1模型的独立验证日期B1诊断，具体数值见DIAGNOSTIC_FINDINGS.md。

最终采用完整事务载荷的单文件完成快照，包含交互、反馈、提交回合和计数，并带摘要。导出轨迹须逐条匹配快照；其他副本差异单列。普通MAPPO与Pareto-MAPPO独立保留。
