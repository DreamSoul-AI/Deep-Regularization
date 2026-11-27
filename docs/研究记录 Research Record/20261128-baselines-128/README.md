# 128维 Ranking Baseline 证据

对应 [完整比较记录](<../2026年11月28日 Ranking Baseline 全量对照.md>)。这是固定参数的合成数据 ranking 迁移实验，不能替代各论文完整调参后的主表。

- `verified_results.jsonl`：816条成功记录，含每折128维原始 importance、指标与Top-k；原方案三条方法线来自此前已核验的复现。
- `summary.json` / `table.md`：按四任务各六折汇总，失败和缺折不生成完整 Macro。
- `cases.json`：每折真实支持索引及本地 `.npz` 的SHA-256；与候选一相同数据配对。
- `registry.json`：34条已完成方法线，以及 mRMR、LassoFlexNet 两个未完成项的原因。
- `manifest.json`：固定来源、包版本、设备、来源快照与最终复跑脚本 hash，以及执行差异。
- `failures.json`：保留失败重试信息；修复后成功的折不重复计数。
- `compatibility.json`：旧 NN、FSNet、E2E-FS 的适配补丁。STG collections 别名、SHAP形状及Keras兼容处理见运行脚本和正文。
- `localized-solver-verification.json`：完整首折的作者求解器/分块求解器对照。
- `nn-fit-audit.json`：冻结旧归因骨干的预测质量诊断，没有选后重训练或据此调参。
- `verification.json`：独立指标校验、适配器核验及剩余限制。

不依赖 PyTorch、NumPy 或 GPU 的指标校验：

```powershell
python experiments/verify_baseline_evidence.py "docs/研究记录 Research Record/20261128-baselines-128"
```

它核验 score→指标→汇总的一致性，不证明每个训练实现正确。Checkpoint、完整训练日志、失败的早期数值门限检查和中断试跑留在本地 `.tmp/all-baselines-20261128/`；数据与原方案 checkpoint 位于 `.tmp/ranking-taylor-fisher/reproduction-128-v2/`，没有把大文件上传仓库。

完整运行需先按 [固定来源准备脚本](../../../experiments/prepare_baseline_sources.py) 建立独立来源目录，并按 [依赖清单](../../../experiments/requirements-baselines.txt) 准备环境。CAE 使用另一个安装 concrete-autoencoder 与 tf-keras 的解释器。

```powershell
python experiments/prepare_baseline_sources.py --output .tmp/baseline-sources
python experiments/run_baselines.py --assets .tmp/baseline-sources --cases .tmp/ranking-taylor-fisher/reproduction-128-v2 --output .tmp/baseline-repeat --cae-python .tmp/cae-env/Scripts/python.exe --device cuda
python experiments/summarize_baselines.py --results .tmp/baseline-repeat --cases .tmp/ranking-taylor-fisher/reproduction-128-v2 --ours .tmp/ranking-taylor-fisher/reproduction-128-v2/results.jsonl --output .tmp/baseline-repeat-summary
```

来源准备目标须为空目录。运行入口默认CPU，只有明确指定 `--device cuda` 时才让 LassoNet/Attention Maps 使用GPU；不自动安装系统工具。保留没有取得源码或依赖不满足的状态，因此完整队列遇到这些项目会返回非零状态，并不代表成功方法的结果失效。
