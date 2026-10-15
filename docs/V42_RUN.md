# v4.2：自适应 DT 区间与真实日期配对实验

本版保留 v4.1 安全执行、延迟感知 MPC、普通 MAPPO 和 Pareto-MAPPO；新增自适应区间为待验证研究分支，不预设优于固定区间。

主训练入口继续默认固定区间 residual；自适应分支须显式指定 `--config configs/v42_adaptive_ieee33.json`。该选择保留原方案，未把负结果包装为默认改进。

## 方法与接口

- 训练日期内五折交叉拟合，产生样本外残差；用固定正则 100 的对数绝对误差回归拟合尺度。
- 点预测与固定区间 residual 完全相同。校准标签只决定分位数，不参与点预测或尺度模型拟合。
- 25 个日期块分别取全天所有 9 输出的最大标准化误差，按有限样本秩计算 90% 名义区间。日期交换性未证实，分布漂移或策略改变时只报告实测覆盖。
- 80 维公开观察接口不变；54:63 为根据当前原始公开观察计算的实际半宽，再传给安全层。模型文件 halfwidth 为参考宽度，不能作为 adaptive 在线宽度。
- 原 constant 模型兼容；旧在线再校准器明确拒绝 adaptive，须重新离线校准。

## 配对实验

```bash
python -m vpp_research.study_v42 --data-root data/real --calibration-days 25 --eval-days 29 --control-days 29 --seed 42000 --workers 6 --output runs/v42_dt
```

输出包括源数据 SHA256 与日期协议、训练/校准/测试观测及离线标签、三个模型、逐日期逐步闭环结果、失败原因和配对成本统计。`--workers` 控制日期任务并行数，同一协议可断点续跑；改变协议须新目录。没有真实数据即报错。种子不代替独立日期。

真实输入为 GB 国家能源与碳曲线，非 IEEE33 实测；设备和通信参数仍是假设。未取得同地区真实会话及 DR 参数，因此本实验 EV/DR 显式关闭。3 月日期此前参与开发，是回顾性评估，不是全新确认性测试。配对成本统计仅针对双方完成的日期，必须同时报告失败对数；时间相关性下 t 区间只能作描述。

完整结果报告可重新生成：`python -m vpp_research.report_v42 runs/v42_dt`。

## 双 MAPPO 与统一基线入口

```bash
python run_v42.py --data-root data/real --seeds 1,2,3,4,5 --episodes 120 --eval-days 29 --selection-days 2 --dt-calibration-days 25 --workers 1 --output runs/v42_full_120
```

请先审阅 DT 比较结果再安排大预算训练；这个命令是可用入口，不表示已经完成训练。所有默认基线继续沿用统一预算和报告系统，算法名称与可用参数见 `python run_v42.py --help`。默认配置为固定区间；自适应分支作为消融候选保留。不要把实验分支的接口验收当作算法优越性证据。

## 验收

```bash
python -m unittest validation.test_v42 validation.test_v41 -v
```

覆盖校准标签隔离、相同点预测、有限样本秩、日期泄漏拒绝、自适应实际宽度接口及 v4.1 安全回归。单元测试中的合成数组只验证程序性质，不作为真实实验结论。

## 先做真实数据端到端验收

以下命令只验收完整流程，不用来判断性能或收敛。普通 MAPPO 和 Pareto-MAPPO 各训练 2 回合；保留完整 24 小时日，不截短物理时域。

```bash
python run_v42.py --data-root data/real --seeds 42 --episodes 2 --eval-days 1 --selection-days 1 --dt-calibration-days 25 --validation-every 0 --eval-preferences "0.2,0.3,0.5" --methods ordinary pareto --workers 1 --output runs/v42_acceptance
python scripts/verify_v42_run.py runs/v42_acceptance
```

`--validation-every 0` 关闭训练过程中的周期选模评估；训练后的独立验证和测试仍执行。正式性能实验不要直接沿用此小预算。验收脚本要求两个算法都存在、完整成功、训练账本与已提交步数一致、测试步数达到协议要求；失败时退出码非零。结果写入 `acceptance_summary.json`。

运行期间不要修改训练源代码。实验协议会对实现做指纹校验，变更后拒绝混用旧结果。若修改了配置、数据或代码，请换输出目录；仅同一协议的中断恢复使用 `--resume`。
