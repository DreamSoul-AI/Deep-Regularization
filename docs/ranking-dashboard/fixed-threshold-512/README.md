# 固定软阈值的左输入对照

本轮在512维四个任务、每任务六折，统一使用原方法软阈值 `clip(C/(A+1e-8), 1e-8, 0.5)/ρ`。A 是第一层对应特征的列范数。q 是 Gate 幅度、一阶、纯二阶或一阶加二阶评分。在比较 q 与 Aq 时都使用原始评分尺度。结果入口为 [实验看板](<../../实验看板 Deep Regularization.md>)。

- `readout/` 复用原训练（z 步输入 s+u，初始ρ50），固定最终模型，同时读出 q、Aq、soft(q) 和 soft(Aq)
- `score/` 在训练 z 步直接用 q 或 Aq，右边仍按原方法设阈值，最终按同一左边公式读出原分数及其 soft 版本
- `weighted_consensus/` 在训练 z 步用 A(s+u)，与原训练的 s+u 比较，同样保存所有评分读出
- `uniform-readout/` 固定原训练和最终左边 Aq，配对比较原阈值 C/(ρA)、共用阈值 C/ρ 和直接排名。`comparison.json` 包含逐折正分召回、非零数及 Top-k 集合变化。脚本 `experiments/rescore_uniform_threshold.py` 校验原 checkpoint 与此前两种读出后，计算共用阈值结果。

训练中的 A 沿用原方法在每个 ADMM epoch 开始时计算。梯度评分每10个 epoch 在训练样本上刷新，Gate 在 z 步用当前幅度。各训练分支保留相同的 cubic 公共缩放、对偶更新和ρ更新规则。改用非负 q 的训练是对 z 输入的直接消融，含义不同于把 q 用作共识分母。

`verified_results.jsonl` 保存每折排名、标准 Top-k 和仅计正分的召回。新对照表只用正分填 Top-k，避免全零或零分并列时被列顺序碰巧算作找回。`manifest.json` 包含训练参数、训练成功或失败、最终 z 的非零数和原始残差。原 ADMM 训练模型如使用 soft 输入再乘 A，实际权重为该模型在对应轮的 A；跨训练模型的 A 与ρ会随轨迹变化。

训练入口为 `experiments/ranking_fixed_threshold.py`。每个子目录均可运行 `python experiments/verify_baseline_evidence.py <子目录>` 核对标准排名指标。全量 checkpoint 留在 `.tmp/ranking-fixed-threshold-512/`。
