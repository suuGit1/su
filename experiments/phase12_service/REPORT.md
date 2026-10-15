# 周期内服务分配复验

记录120/120，审计问题0。仅开发集，未评估封存确认集。

| 条件/修订/任务 | 完成/请求 | 日均成本 | 日均碳 | 日均备用 | 无效备用步 |
|---|---:|---:|---:|---:|---:|
| c3/nominal/service_risk/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/nominal/service_value/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/uplink/service_risk/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/uplink/service_value/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/compute/service_risk/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/compute/service_value/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/combined/service_risk/balanced | 24/24 | 1295.940 | 4918.085 | 3927.601 | 0 |
| c3/combined/service_value/balanced | 24/24 | 1298.126 | 4920.091 | 3812.578 | 0 |
| tasks/combined/service_value/economic | 8/8 | 1271.815 | 4921.426 | 3936.580 | 0 |
| tasks/combined/service_value/carbon | 8/8 | 1303.006 | 4917.807 | 3955.907 | 0 |
| tasks/combined/service_value/reserve | 8/8 | 1303.006 | 4917.807 | 3955.907 | 0 |

| 配对比较（左减右） | 成本差 | 碳差 | 有效备用差 |
|---|---:|---:|---:|
| nominal/service_risk-v1_uniform | -0.388 | 5.734 | -71.264 |
| nominal/service_risk-v1_active_equal | 0.000 | 0.000 | -0.000 |
| nominal/service_risk-v1_joint_risk | -0.388 | 5.734 | -71.264 |
| nominal/service_value-v1_uniform | -0.388 | 5.734 | -71.264 |
| nominal/service_value-v1_active_equal | 0.000 | 0.000 | -0.000 |
| nominal/service_value-v1_joint_risk | -0.388 | 5.734 | -71.264 |
| nominal/service_value-service_risk | 0.000 | 0.000 | 0.000 |
| uplink/service_risk-v1_uniform | -73.638 | 8.094 | -10.519 |
| uplink/service_risk-v1_active_equal | 0.000 | 0.000 | -0.000 |
| uplink/service_risk-v1_joint_risk | -46.409 | 5.504 | -196.340 |
| uplink/service_value-v1_uniform | -73.638 | 8.094 | -10.519 |
| uplink/service_value-v1_active_equal | 0.000 | 0.000 | -0.000 |
| uplink/service_value-v1_joint_risk | -46.409 | 5.504 | -196.340 |
| uplink/service_value-service_risk | 0.000 | 0.000 | 0.000 |
| compute/service_risk-v1_uniform | -73.638 | 8.094 | -10.519 |
| compute/service_risk-v1_active_equal | 0.000 | 0.000 | -0.000 |
| compute/service_risk-v1_joint_risk | -38.913 | 5.066 | -153.124 |
| compute/service_value-v1_uniform | -73.638 | 8.094 | -10.519 |
| compute/service_value-v1_active_equal | 0.000 | 0.000 | -0.000 |
| compute/service_value-v1_joint_risk | -38.913 | 5.066 | -153.124 |
| compute/service_value-service_risk | 0.000 | 0.000 | 0.000 |
| combined/service_risk-v1_uniform | 0.325 | -0.794 | 23.193 |
| combined/service_risk-v1_active_equal | -48.953 | 1.746 | 107.489 |
| combined/service_risk-v1_joint_risk | 53.548 | -8.623 | -223.509 |
| combined/service_value-v1_uniform | 2.511 | 1.212 | -91.830 |
| combined/service_value-v1_active_equal | -46.767 | 3.752 | -7.534 |
| combined/service_value-v1_joint_risk | 55.734 | -6.617 | -338.532 |
| combined/service_value-service_risk | 2.186 | 2.006 | -115.024 |
| task/economic-balanced | -36.686 | 4.218 | -12.418 |
| task/carbon-balanced | -5.494 | 0.598 | 6.909 |
| task/reserve-balanced | -5.494 | 0.598 | 6.909 |

所有配对日期、剔除原因与失败计数保存在summary.json。未确认备用的配对不进入有效备用差。成本与碳越低越好，备用越高越好；不得跨指标挑选任务成功标准。
