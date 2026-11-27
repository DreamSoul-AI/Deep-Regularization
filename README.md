# Deep-Regularization

当前实验状态与完整结果统一从 [Deep Regularization 实验看板](<docs/实验看板 Deep Regularization.md>) 查看。看板区分512维全量配对运行、原仓库默认及历史均值，未完成项明确保留；下列日期记录用于追溯。

## Research notes

- [XOR / Ring 研究线程总结（2026-09-20）](docs/research/20260920-xor-ring-thread/README.md)
- [会议汇报：承接 09/06 讨论](docs/research/20260920-xor-ring-thread/MEETING_REPORT.md)
- [128 维全部 baseline 对照](docs/research/20260920-xor-ring-thread/baseline_128_table.md)

上述2026年9月历史总结包含五项候选、高维 XOR / Ring、两条 legacy 路径及后续诊断的结论、图表和来源快照，当时仅合入文档与证据。11月新增授权后的代码和实验状态以当前看板及 REPO.md 为准，不据此改写旧暂停记录。

可运行以下命令核对便携证据包，无需训练环境：

```sh
python docs/research/20260920-xor-ring-thread/verify_bundle.py
```

## Repository maintenance

See [repository conventions](REPO.md) and the [engineering document index](docs/README.md) for maintenance.

## Deep Regularization 研究计划

- [研究计划与实验协议](<docs/研究计划 Research Plan Deep Regularization.md>)
- [输入门控、LassoNet 与 ADMM 方法推导](<docs/调研 Search/通用调研 General Search/通用调研 General Search Deep Regularization 输入门控与 LassoNet.md>)
- [四候选文献、baseline 与 benchmark 对照](<docs/调研 Search/文献调研 Literature Search/文献调研 Literature Search Deep Regularization.md>)

当前计划包含四个候选，以及 feature ranking、LassoFlexNet 协议下游效果两块正式实验。11月新增授权已实施候选一、baseline 适配和有效 gate 报告参照；128维已完成，512维全量结果持续汇总于当前看板。其他候选、搜索与下游不因本次排名授权自动启动。
