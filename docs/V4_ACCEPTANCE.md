# v4 功能验收复现

验收使用一个种子和极小训练预算，仅检查训练—选择—测试—报告闭环、费用口径及预算审计。不能据此证明收敛、统计显著性或优于 MPC。实际结果见 `experiments/v4/verification_summary.json`。

从项目根目录运行，Python 依赖同 `docs/V4_RUN.md`。

```bash
python -X utf8 -m unittest validation.test_v3 validation.test_v4 -q
python -X utf8 scripts/prepare_v3_acceptance_data.py --output runs/v4_acceptance_data
python run_v4.py --data-root runs/v4_acceptance_data --seeds 1 --episodes 4 --rollout-episodes 2 --validation-every 4 --eval-days 1 --selection-days 1 --dt-train-days 2 --dt-calibration-days 9 --eval-preferences "0.2,0.3,0.5" --workers 1 --output runs/v4_seven_final
python run_v4.py --data-root data/real --seeds 1 --episodes 2 --rollout-episodes 2 --validation-every 2 --eval-days 1 --selection-days 1 --dt-train-days 2 --dt-calibration-days 9 --eval-preferences "0.2,0.3,0.5" --methods mpc ordinary pareto --workers 1 --output runs/v4_full_day_final
python -X utf8 scripts/report_v4.py runs/v4_seven_final
python -X utf8 scripts/report_v4.py runs/v4_full_day_final
```

短链路保留真实曲线每日前两小时，完整日使用原始24小时数据。两者都不是现场电网试验。重新运行应换新的输出路径；已完成结果如需续跑则保持原参数并加 `--resume`。完整包中保留现有验收结果时，不要覆盖它们。

报告路径相对于项目根目录；重新生成图表时请在根目录执行。普通 MAPPO、Pareto-MAPPO 和其他学习对照共用80维公开观察，MPC的54维接口限制见运行说明。

## 2026-10-08 实际结果

15项回归通过；七方法短链路均完成，五种学习方法各8个训练交互，80次定期验证全部成功。完整日普通/Pareto各48个训练交互，三方法各24个测试步，192次定期验证全部成功。费用分项复算、轨迹和调用预算审计通过。

| 完整日方法 | AC违例 | 备用未确认步 | 区间证书成立步 | 严格可行日 |
|---|---:|---:|---:|---:|
| MPC | 0 | 0 | 0/24 | 1/1 |
| Ordinary MAPPO | 0 | 0 | 0/24 | 1/1 |
| Pareto-MAPPO | 0 | 1 | 0/24 | 0/1 |

**运行完成不等于安全全部通过。** Pareto的失败日保留在结果和严格HV中。功能验收已完成，算法性能和鲁棒安全验收尚未完成；需要分析备用确认失败、区间证书不成立原因，并补齐延迟感知MPC后开展长预算多种子比较。
