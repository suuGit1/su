# DT-C3-VPP v4

完整运行、预算、恢复和报告说明见 [docs/V4_RUN.md](docs/V4_RUN.md)。

```powershell
python run_v4.py --data-root data/real --methods mpc ordinary pareto --seeds 1 --episodes 4 --rollout-episodes 2 --validation-every 4 --eval-days 1 --selection-days 1 --dt-train-days 2 --dt-calibration-days 9 --workers 1 --output runs/v4_smoke
```

普通MAPPO保留官方更新器。v4需要新输出目录；代码升级不代表已经取得统计显著的算法优势。
