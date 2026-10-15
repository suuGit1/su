# 协调器修订复验

记录120/120，审计问题0。仅开发集，未评估封存确认集。

| 条件/修订/任务 | 完成/请求 | 日均成本 | 日均碳 | 日均备用 | 无效备用步 |
|---|---:|---:|---:|---:|---:|
| c3/nominal/mask_only/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/nominal/mapped/balanced | 8/8 | 1245.597 | 4922.032 | 4025.651 | 0 |
| c3/uplink/mask_only/balanced | 8/8 | 1346.028 | 4911.819 | 4104.639 | 0 |
| c3/uplink/mapped/balanced | 8/8 | 1273.706 | 4920.602 | 3839.229 | 0 |
| c3/compute/mask_only/balanced | 8/8 | 1346.032 | 4911.808 | 4102.694 | 0 |
| c3/compute/mapped/balanced | 8/8 | 1244.992 | 4924.764 | 3798.987 | 0 |
| c3/combined/mask_only/balanced | 24/24 | 1279.871 | 4920.865 | 4013.888 | 0 |
| c3/combined/mapped/balanced | 24/24 | 1250.937 | 4923.326 | 4009.420 | 0 |
| tasks/combined/mapped/economic | 8/8 | 1319.786 | 4915.495 | 4142.261 | 0 |
| tasks/combined/mapped/carbon | 8/8 | 1301.686 | 4916.232 | 4172.211 | 0 |
| tasks/combined/mapped/reserve | 8/8 | 1295.308 | 4920.149 | 3960.578 | 0 |

| 配对比较（左减右） | 成本差 | 碳差 | 有效备用差 |
|---|---:|---:|---:|
| nominal/mask_only-v1_uniform | -0.388 | 5.734 | -71.264 |
| nominal/mask_only-v1_active_equal | 0.000 | 0.000 | 0.000 |
| nominal/mask_only-v1_joint_risk | -0.388 | 5.734 | -71.264 |
| nominal/mapped-v1_uniform | -0.388 | 5.734 | -71.264 |
| nominal/mapped-v1_active_equal | 0.000 | 0.000 | 0.000 |
| nominal/mapped-v1_joint_risk | -0.388 | 5.734 | -71.264 |
| nominal/mapped-mask_only | 0.000 | 0.000 | 0.000 |
| uplink/mask_only-v1_uniform | 26.793 | -2.118 | 68.469 |
| uplink/mask_only-v1_active_equal | 100.432 | -10.213 | 78.988 |
| uplink/mask_only-v1_joint_risk | 54.023 | -4.708 | -117.351 |
| uplink/mapped-v1_uniform | -45.529 | 6.665 | -196.941 |
| uplink/mapped-v1_active_equal | 28.110 | -1.430 | -186.422 |
| uplink/mapped-v1_joint_risk | -18.299 | 4.075 | -382.762 |
| uplink/mapped-mask_only | -72.322 | 8.783 | -265.410 |
| compute/mask_only-v1_uniform | 26.797 | -2.130 | 66.524 |
| compute/mask_only-v1_active_equal | 100.436 | -10.224 | 77.043 |
| compute/mask_only-v1_joint_risk | 61.523 | -5.158 | -76.081 |
| compute/mapped-v1_uniform | -74.243 | 10.826 | -237.183 |
| compute/mapped-v1_active_equal | -0.605 | 2.732 | -226.664 |
| compute/mapped-v1_joint_risk | -39.518 | 7.798 | -379.788 |
| compute/mapped-mask_only | -101.041 | 12.956 | -303.707 |
| combined/mask_only-v1_uniform | -15.744 | 1.987 | 109.479 |
| combined/mask_only-v1_active_equal | -65.022 | 4.526 | 193.775 |
| combined/mask_only-v1_joint_risk | 37.480 | -5.842 | -137.223 |
| combined/mapped-v1_uniform | -44.678 | 4.448 | 105.012 |
| combined/mapped-v1_active_equal | -93.956 | 6.987 | 189.308 |
| combined/mapped-v1_joint_risk | 8.545 | -3.381 | -141.690 |
| combined/mapped-mask_only | -28.934 | 2.461 | -4.468 |
| task/economic-balanced | -17.735 | 3.962 | 101.848 |
| task/carbon-balanced | -35.835 | 4.699 | 131.798 |
| task/reserve-balanced | -42.213 | 8.616 | -79.836 |

所有配对日期、剔除原因与失败计数保存在summary.json。未确认备用的配对不进入有效备用差。成本与碳越低越好，备用越高越好；不得跨指标挑选任务成功标准。
