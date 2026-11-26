# Deep-Regularization

## Research notes

- [XOR / Ring 研究线程总结（2026-09-20）](docs/research/20260920-xor-ring-thread/README.md)
- [会议汇报：承接 09/06 讨论](docs/research/20260920-xor-ring-thread/MEETING_REPORT.md)
- [128 维全部 baseline 对照](docs/research/20260920-xor-ring-thread/baseline_128_table.md)

研究总结包含五项候选、高维 XOR / Ring、两条 legacy 路径及后续诊断的结论、图表和来源快照。此次合入仅涉及文档与证据；模型实现和默认配置未变。实验自 2026-09-18 暂停。

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

当前计划包含四个候选，以及 feature ranking、LassoFlexNet 协议下游效果两块正式实验。文档中的实现、参数与求解器依赖仍待落实，实验暂停状态保持原约定。
