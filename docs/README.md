# 研发文档导航

Research evidence and reproducibility notes. Consult the retained research index and existing execution restrictions below.

| 任务 | 入口 |
| --- | --- |
| 项目说明与使用 | [README.md](../README.md) |
| 仓库专用工作约定 | [REPO.md](../REPO.md) |
| 调研与证据 | [调研与证据](<调研 Search/README.md>) |
| 开发与验证 | [开发与验证](<开发 Development/README.md>) |

现行设计与计划从对应入口维护。历史开发、运行证据和研究素材按原有记录定位，保留实际日期、版本及证据来源。

## Deep Regularization 研究计划

- [研究计划与实验协议](<研究计划 Research Plan Deep Regularization.md>)
- [输入门控、LassoNet 与 ADMM 方法推导](<调研 Search/通用调研 General Search/通用调研 General Search Deep Regularization 输入门控与 LassoNet.md>)
- [四候选文献、baseline 与 benchmark 对照](<调研 Search/文献调研 Literature Search/文献调研 Literature Search Deep Regularization.md>)

当前计划包含四个候选，以及 feature ranking、LassoFlexNet 协议下游效果两块正式实验。按11月27日至28日新增授权，已实施候选一并开展128维 ranking 配对复现与全 baseline 比较；其他候选、维度网格与下游实验仍按原范围待执行。

- [候选一 ranking 初测与实现记录](<研究记录 Research Record/2026年11月27日 候选一 Ranking 初测.md>)
- [原方案500轮复现与四任务ranking总表](<研究记录 Research Record/2026年11月28日 原方案复现与候选一 Ranking 对照.md>)
- [全部 ranking baseline、论文新增方法与运行限制](<研究记录 Research Record/2026年11月28日 Ranking Baseline 全量对照.md>)
- [mRMR 修复、LassoFlexNet 本地实现与512维版本复核](<研究记录 Research Record/2026年11月28日 Baseline 修复与512维复核.md>)
- [保持方法的有效 gate 512维 ranking 对照](<研究记录 Research Record/2026年11月28日 有效 gate 的512维 Ranking 对照.md>)

后续新增授权已完成 mRMR、本地 LassoFlexNet 数值二分类实现，以及512维 XOR/Ring 的固定设置比较。后者使用公开 raw-gate 500/120轮参照；飞书10月报告的有效 gate GRN-FS 及11月 A—L 源码尚未取得，不能以公开快照成功替代其高维失败复核。候选二至四与下游仍未执行。
