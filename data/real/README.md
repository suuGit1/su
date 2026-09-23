# 统一真实数据目录

默认调用 `gb/profiles`：2019 年英国能源日曲线与同地区碳强度，UTC 对齐。

|数据|已取得|程序用途|
|---|---|---|
|OPSD 负荷、PV、风电、电价|GB 2019年1–3月有效曲线|训练31天、DT校准27天、独立测试29天|
|NESO 碳强度|上述日期的半小时实际核算值|小时均值用于核算，上小时值作为滞后观察代理|
|CPUC/SCE ELRP A.4 VPP DR|55条小时记录，其中35条有效事件、10条负响应|`--dr-mode sce-derated`，仅使用训练期响应分位数降额|
|DE OPSD 旧处理结果|保存在 `archive/de`|仅历史参考，不用于默认GB三目标实验|
|真实EV完整会话|尚未取得|默认明确关闭；后续 `--ev-sessions 文件.json` 接入|
|硬件能耗和通信实测|尚未取得|参数继续明确标注为仿真假设|

默认资源配置关闭 EV、DR，不自动补充合成车辆或DR需求。SCE 数据来自加州2022年，不能称为英国2019年的同期实测；训练响应 q10 为负，保守截断后削减容量为0，这是数据结果，不能人为调正以制造调控能力。EV/DR模块和Agent接口仍保留。

`gb/profiles/manifest.json` 记录缩放、单位、来源和哈希。国家级 MW 曲线缩放到 VPP kW 规模，不是 IEEE33 馈线实测。储能、线路额定、价格外的服务成本和 C3 设备参数仍需研究假设。

原始文件位于完整数据包的 `raw/opsd`、`raw/neso` 和 `dr/sce/source.xlsx`。仓库自带可直接运行的处理后 CSV、来源清单和 DR 事件 JSON；原始压缩 CSV 与工作簿在数据包内，不必下载即可使用处理后数据训练。

校验：`python scripts/prepare_real_data.py --verify-only`

从原始文件重建到新目录：`python scripts/prepare_real_data.py --output data/rebuilt`

原始来源：

- https://data.open-power-system-data.org/time_series/2020-10-06/
- https://api.carbonintensity.org.uk/intensity/date/
- https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/energy-division/documents/demand-response/emergency-load-reduction-program/elrp-2022-program-data/sce-elrp-hourly-with-lip-values.xlsx

没有可用数据的日期被整日剔除，不插值、不重复日期补足测试天数。当前总计87个完整日，测试缺少3月30/31日，因此 `--eval-days 30` 会明确报错；扩展真实数据后该参数可继续增大。
