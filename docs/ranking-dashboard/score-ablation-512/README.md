# 512维 gate 评分对照证据

完成结果与原方案并列展示在 [当前实验看板](<../../实验看板 Deep Regularization.md>)。本目录只补六条新方法的逐折证据，原方案及已有梯度评分读取 [此前512维汇总](../512/summary.json)。

`manifest.json` 保存同批24个数据文件的哈希、固定训练配置、代码哈希、分片和核验结果。`verified_results.jsonl` 保存从训练 checkpoint 读出的最终 gate 幅度和逐折指标。`summary.json` 保存四任务汇总，`cases.json` 的坐标和数据哈希与原512维看板一致。

标准库核验命令为 `python experiments/verify_baseline_evidence.py docs/ranking-dashboard/score-ablation-512`。复跑入口为 `experiments/ranking_score_ablation.py`。本地训练 checkpoint 和日志位于 `.tmp/ranking-score-ablation-512/`。
