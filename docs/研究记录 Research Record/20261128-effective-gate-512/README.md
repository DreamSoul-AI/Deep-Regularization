# 有效 gate 的512维排名证据

对应 [比较记录](<../2026年11月28日 有效 gate 的512维 Ranking 对照.md>)。

- `verified_results.jsonl`：有效 gate、现有一阶＋Fisher和LassoNet，各12折，共36条512维分数
- `cases.json`：数据哈希和真实支持，沿用此前同一批配对数据
- `summary.json` / `table.md`：两个任务的排名指标和均值
- `manifest.json`：新参照的固定报告参数、预处理和源码哈希
- `checkpoint-verification.json`：12个新模型重载和1440次原生近端认证结果
- `comparison-provenance.json`：现有两个方法的记录来源及整体文件哈希，明确本轮只新增参照方法

标准库核验：`python experiments/verify_baseline_evidence.py "docs/研究记录 Research Record/20261128-effective-gate-512"`。该证据包不包含模型及完整数据；本地重载命令见比较记录。
