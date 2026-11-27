# 512维当前排名证据

人读入口统一为 [Deep Regularization 实验看板](<../../实验看板 Deep Regularization.md>)，不另建日期总表。

- `verified_results.jsonl`、`summary.json`、`cases.json`：当前逐折512维分数、按四任务汇总及配对数据身份。未完成项不生成Macro
- `sources.json`、`protocol.json`：原始结果分片哈希、固定commit、方法范围、软件环境、内存执行适配及未覆盖范围
- `failures.json`：模型/求解器失败记录；本机有意停止的内存超大试跑及分片调度保留在 `.tmp/` 的单独操作记录，不当成有效成绩
- `public-wrapper-contract.json`：原ρ200包装器首折与新公开默认入口的完整分数逐元素一致
- `raw-snapshot-verification.json`：ρ50配方四任务的24个原训练参数对齐、96个模型重载和72个评分矩重算
- `effective-*-verification.json`：有效 gate 的原两个任务和新增混合任务，共24模型及2880次原生近端认证
- `flex-*-verification.json`：LassoFlexNet本地文本变体的24个checkpoint重载、训练分箱和路径选模核验
- `attention-gradient-contract.json`：实际上游FT模型关闭dropout时，完整/分块目标与梯度一致。开启dropout不声明逐位同轨迹
- `lassonet-device-audit.json`：本机旧原生LassoNet自动使用CUDA的设备更正依据，原始分数保留；新显式CPU/GPU记录不套用此更正
- `rho-reference-comparison.json`：ρ50与ρ200的Top-k集合、分数向量差异，避免把相同恢复率误当成重复实验
- 全批模型读出、Localized Lasso各次线性方程残差和Attention有效batch等最终核验另存 `model-verification.json`

标准库核验：`python experiments/verify_baseline_evidence.py docs/ranking-dashboard/512`。

刷新当前结果：`python experiments/update_ranking_dashboard.py`。模型、完整日志、旧失败试跑、原始环境和数据仅留在本机 `.tmp/`，不把大型训练原件加入Git。
