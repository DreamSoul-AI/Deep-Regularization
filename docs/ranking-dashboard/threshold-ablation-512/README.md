# 512维软阈值对照证据

四种特征评分分别比较是否乘第一层列范数 A，以及训练时是否使用软阈值。四任务均为同一批512维数据、六折、500轮和初始ρ50。各对照的最终排序使用相同评分公式，逐折也保存只用训练后 gate 幅度排名的指标，用来衔接上一轮。

无软阈值时直接将共识中心传给原有公共缩放，保留网络损失、ADMM 对偶更新和全局ρ调节。训练过程与评分无关，因此每折只训练一份模型，按八种评分各自读出。使用软阈值时按评分逐特征设置阈值，复用同设置的七个已核验模型变体，并补跑 `gate` 本身调阈值的一组。

`summary.json` 保存16种读出在四个任务上的平均。`verified_results.jsonl` 保存384个配对结果。`manifest.json` 包含运行参数、文件哈希、已有模型身份核验、无软阈值更新核验、统一 gate 读出均值，以及相同八种读出在原列范数方案的24个训练模型上的对照。完整结果入口是 [实验看板](<../../实验看板 Deep Regularization.md>)。

指标核验用 `python experiments/verify_baseline_evidence.py docs/ranking-dashboard/threshold-ablation-512`。复跑入口是 `experiments/ranking_threshold_ablation.py`。训练原件留在原实验的 `.tmp/` 目录。
