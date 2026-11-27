# 默认评分与最终排名

当前 [实验看板](<../../实验看板 Deep Regularization.md>) 使用训练终点的阈值评分排名。训练时使用软阈值与否，排名公式一致。列范数方案按 A 排，梯度组按对应删除评分排，使用乘积的一组按乘积排。仅 gate 统一阈值对照按 |s| 排。外部 baseline 与有效 gate 参照沿用各自评分。

`128/` 和 `512/` 保存由已训练 checkpoint 重算的逐折 score、汇总、数据身份和源码哈希。512维此外保存每个使用软阈值的模型分别按 q 和 Aq 排名的配对结果。q 是不乘第一层列范数 A 的基础评分。每对评分读取同一个 checkpoint。关闭软阈值的配对读出保存在 [软阈值对照](../threshold-ablation-512/README.md)。

128维旧结果保存在 [原始实验记录](<../../研究记录 Research Record/20261128-baseline-repairs/128/table.md>)。512维统一用 |s| 排名的历史结果与外部 baseline 汇总保存在 [旧主表证据](../512/summary.json) 和 [评分消融证据](../score-ablation-512/summary.json)。

本轮复用已训练模型。源数据与 checkpoint 的标准化统计量及原始 gate 向量逐折验证。复算入口为 `experiments/rescore_ranking.py`，标准库核验为 `python experiments/verify_baseline_evidence.py docs/ranking-dashboard/score-readout/512`，128维替换末尾目录。
