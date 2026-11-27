# Baseline 修复与512维证据

对应 [结果与实现记录](<../2026年11月28日 Baseline 修复与512维复核.md>)。

- `128/`：36方法、864分数向量；`verified_results.jsonl` 只保存新增48条，`record_sources.json` 明确引用旧816条，避免重复归档。旧失败状态为历史，新增完成状态见本目录汇总。
- `512/`：7方法、84分数向量，仅 XOR/Ring；每折包含真实支持及数据哈希。公开 raw-gate 对齐结果与 manifest 单独保存。
- 两个目录的 `flex-manifest.json` 和 `flex-checkpoint-verification.json` 记录 v2 本地可行实现配置及实际模型重载核验。
- `native-contracts.json`、`mrmr-prefix-verification.json`、`pymrmre-compatibility.json` 记录原生求解器的信号测试、mRMR Top-k 前缀核验和两个包装接口修复。
- `flex-v2-literal-verification.json` 是印刷算法2违反 hierarchy 的单折诊断，未混入主表。
- `source-hashes.json` 为执行/汇总源码的换行归一化哈希。训练时原始哈希另在 manifest 保留。

标准库指标核验入口：`python experiments/verify_baseline_evidence.py "docs/研究记录 Research Record/20261128-baseline-repairs/128"`；末尾改为512可核验另一表。

Checkpoint、数据、安装包、环境、原始飞书抓取及完整日志只留本机 `.tmp/baseline-repairs-20261128/`。本目录是可携带的排名及验证证据，不包含独立训练所需全部原件。
