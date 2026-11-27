# 512 维正则强度与特征排名

本轮固定原 ADMM 训练，扫描 `C = 0.0125, 0.025, 0.05, 0.1`。使用 512 维、数据种子 1–5、四个任务、每任务六折。`C=0.05` 复用已验证的模型，其余三档各训练 120 折。每个模型按 `Aq` 与 `soft(Aq, C/(rho A))` 分别读出 Gate、一阶、纯二阶及一阶加二阶四种分数，零分不补齐排名。跨方法使用同一份真实特征支持和同一 Top-k 恢复率，其中四个任务的 k 分别为 2、2、4、6。

Random Forest 复用看板的 500 棵树、原随机种子和训练折标准化，用同一批新数据折重跑，按原生重要性排序。正则强度比较和 Random Forest 只做排名。

[实验看板](<../../实验看板 Deep Regularization.md>) · [分组与配对汇总](summary.json) · [逐折核验](verified_folds.jsonl)

## 扩展种子后的基线对照

用种子 1–5 看参数敏感性和确定每种方法的设置。各方法的设置冻结后，在新增的种子 6–10 上做配对比较，报告均值、逐种子差值和胜／平／负。配置可以用开发种子的真实特征恢复率做选择，最终种子的真实特征仅用于评分。四任务等权，所有方法用同一批数据折和同一 Top-k。原生的完整路径排名保留为固定对照。

| 方法 | 扫描的对象 |
| --- | --- |
| 本方法 | 训练正则强度 `C`，其余 ADMM 设置及模型保持一致。原始分数与最终软阈值分数分别报告。 |
| LassoNet | 原包装器已跑完整 λ 路径。保留 `lambda_start=3.2768`、倍率 `1.02` 的原生排名，额外检查较早的起点是否改变退出顺序。另按同一条完整路径的保留特征数标出不同稀疏程度。 |
| 本地 LassoFlexNet 复实现 | 已有 12 个正 λ 的路径与内部验证读出。保留原设置，按预先约定的路径范围做对照。 |
| Localized Lasso | 固定图正则 `lam_net=1`，首先围绕默认 `lam_exc=0.01` 扫排他稀疏项。 |
| STG | 围绕现有 `lam=0.1` 扫 gate 正则，固定 `sigma`。 |
| L1 Logistic Regression | 围绕现有 `C=1` 扫，`C` 越小代表正则越强。 |
| Random Forest | 保持同一训练折的固定参考，沿用特征重要性。 |

LassoNet `feature_importances_` 记录的是特征首次退出路径的 λ。原包装器用最低验证损失的 checkpoint 预测，排名继续使用整条路径。原生排名需要整条路径完成，截断的路径会留下退出强度未知的特征。对照时各方法保留自身正则尺度，以实际稀疏程度解释差异。 [原仓库包装器](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/core.py#L161)

路径入口已在 CPU 上核对种子 1 的 Ring 第 0 折。原起点 3.2768 和较低起点 0.8192 分别在 λ≈3.69 和 λ≈1.32 首次压掉特征，均跑到特征全部退出，两条路径在这一折的 Top-2 恢复率均为 0。这个单折试跑用于检查路径覆盖。多种子 LassoNet 和其他基线的参数扫描排在下一轮。

## 用稠密模型确定扫描范围

先在训练折关闭特征稀疏正则，得到稠密模型。固定其参数和初始化时的阈值尺度，计算每个特征在一次 prox 中刚好归零的临界正则，再用临界值的分位点覆盖从轻度到重度的稀疏区间。正则候选只依赖训练折，真实特征标签用于后续评分。进入实际训练后，记录保留特征数随正则强度的变化，再据此调整扫描范围。

原 ADMM 的训练左输入是 `v = gate + dual`，初始 dual 为零。忽略数值保护时，直接反算得到 `C*_j = rho (A_j + eps) |v_j|`。当前代码把阈值限制在 `0.5/rho`，所以 `rho |v_j| > 0.5` 的特征在这个固定状态下一次 prox 无法归零。`A`、`rho` 和 gate 还会随训练更新，因此临界值用于定范围，最终强度需按实际训练的稀疏度核验。 [训练 prox](../../../src/feature_selection.py#L197)

LassoNet 0.0.22 的 `lambda_start="auto"` 已有类似逻辑。它先做 λ=0 的稠密训练，用分组 prox 探测开始产生零系数的位置，换算优化器步长后从更小的 λ 出发。下一轮优先核对 `auto` 和原固定起点的路径覆盖，再看其他方法如何确定正则扫描上下界。

## 复跑入口

```powershell
python experiments/train_ranking_penalty_sweep.py --penalty 0.025 --seed 1 --output .tmp/ranking-penalty-sweep/0.025-1
python experiments/ranking_penalty_rf_reference.py --seed 1 --output .tmp/ranking-penalty-sweep/rf-1
python experiments/ranking_lassonet_path.py --seed 1 --task ring --fold 0 --device cpu --output .tmp/ranking-penalty-sweep/lassonet-native-cpu-pilot
python experiments/summarize_ranking_penalty_sweep.py --runs .tmp/ranking-penalty-sweep --reference .tmp/ranking-seed-sweep --output docs/ranking-dashboard/penalty-sweep
python experiments/update_ranking_dashboard.py
```
