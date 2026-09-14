# 上游来源

仓库：https://github.com/marlbenchmark/on-policy

固定提交：de66d7a4b23fac2513f56f96f73b3f5cb96695ac。

本目录收录训练所需官方模块及工具依赖，不是完整上游仓库。MIT 许可证见 LICENSE；上游 Git blob SHA 与本地 SHA-256 见 upstream_manifest.json。

唯一源码修改：onpolicy/__init__.py 去除对未收录 envs、runner、scripts 的提前导入，保留版本声明。其余清单文件保存上游原始字节及注释；PPO 数学逻辑、网络、缓存与归一化未修改。根目录 vpp_mappo 提供新的 Gymnasium 空间适配和训练主循环。

论文需引用 MAPPO 原论文和官方实现。执行 python validation/verify_vendor.py 校验固定版本。
