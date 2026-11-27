# 原方法训练的多种子排名对照

固定原训练，在相同模型上配对比较 A × Gate、A × 一阶、A × 纯二阶及 A ×（一阶＋二阶）。结果入口是 [实验看板](<../../实验看板 Deep Regularization.md>)。

使用数据种子 1–5、128/256/512/1024 维、四个合成任务和六折，每个任务每个种子生成1000行。512维按历史生成器生成，128和256维取特征前缀，1024维增加独立噪声列。各维数同任务同种子共享样本、标签和信号。各折按确定性随机置换打乱特征列位置。模型随机种子是0，训练输入标准化、动态阈值及原全局ρ更新保持一致。固定初始ρ50、总500轮和120轮 warm-up，沿用产生60.07%的参照配方。

每折训练一次，梯度评分按原配置在固定的128条训练校准样本上计算。四种读出共用原 soft 阈值及数值保护，排序取正分前 k 个，缺额计作未找回。四任务等权。跨种子稳定性用同种子配对百分点差及胜／平／负描述，六折是一个数据种子内部的测量。旧种子0从新汇总中排除。它的512维生成数据和96条历史读出已逐项核对一致。

[汇总结果](summary.json) 保存逐维数、逐种子、逐任务恢复率和配对差值。[逐折结果](verified_folds.jsonl) 保存每折真实支持、模型与数据哈希、每种评分的正分 Top-k、恢复率和 AP。汇总前重载各模型，核验训练配方、数据、训练折标准化和四种读出。checkpoint 在本机 `.tmp/ranking-seed-sweep/`。

训练入口为 `python experiments/ranking_seed_sweep.py --dimension 512 --seed 1 --output .tmp/ranking-seed-sweep/512-1`。各维数及种子完成后运行 `python experiments/summarize_ranking_seed_sweep.py --runs .tmp/ranking-seed-sweep --output docs/ranking-dashboard/seed-sweep`，再刷新现有看板。
