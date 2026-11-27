# Deep Regularization 实验看板

更新时间：2026-11-28 04:24（北京时间）。这是当前唯一汇总入口；旧日期记录只作方法与证据追溯。

512维全量任务：**38/38条方法线完成**，已核验912/912个逐折结果。范围为 XOR、Ring、Ring+XOR、Ring+XOR+Sum，各1000样本、普通六折，只做ranking，不调参或下游重训。全量指本轮声明的方法清单，不扩到原CLI的attr-t、10模型bootstrap attr-b、普通NN预测或DAG2000维。

## 当前判断

四任务已完成的原仓库默认平均为53.13%，一阶为42.01%，一阶＋Fisher为41.32%。这批固定数据与配方下，换评分没有改善；不能认为二阶总不会差。

全部方法已跑齐。本轮四任务均值最高为 Random Forest，62.50%；结论只适用于本轮固定设置，不替代多种子或充分调参后的比较。

## 怎么读结果

每列是 Top-k 相关特征恢复率：k依次为2、2、4、6。100%表示找回该任务全部真实特征，0%表示一个也没找回。它不是分类准确率。四任务均完成后才计算 Macro；缺折不当作0，也不生成部分任务总排名。

此前的三方法小表只有 XOR/Ring：有效 gate 擅长 Ring、XOR失败；现有一阶＋Fisher在XOR成功、Ring较弱。那张两任务均值不能代表四任务总体，也不能证明只改score有收益，因为骨干、正则求解器和训练预算不同。

## 512维当前完整矩阵

| 方法 | XOR | Ring | Ring+XOR | Ring+XOR+Sum | 四任务 Macro | 完成状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 原仓库512默认（ρ200） | 100.00 | 41.67 | 20.83 | 50.00 | 53.13 | 24/24 完成 |
| 10月有效 gate 报告参照 | 0.00 | 100.00 | 45.83 | 66.67 | 53.13 | 24/24 完成 |
| 固定128配方扩维：列范数（ρ50） | 100.00 | 41.67 | 20.83 | 50.00 | 53.13 | 24/24 完成 |
| 我们的方案：一阶（ρ50） | 100.00 | 16.67 | 4.17 | 47.22 | 42.01 | 24/24 完成 |
| 我们的方案：一阶＋Fisher（ρ50） | 100.00 | 16.67 | 4.17 | 44.44 | 41.32 | 24/24 完成 |
| Random Forest | 0.00 | 100.00 | 75.00 | 75.00 | 62.50 | 24/24 完成 |
| TreeSHAP | 0.00 | 100.00 | 45.83 | 69.44 | 53.82 | 24/24 完成 |
| LassoNet | 83.33 | 0.00 | 16.67 | 36.11 | 34.03 | 24/24 完成 |
| mi | 0.00 | 100.00 | 4.17 | 8.33 | 28.13 | 24/24 完成 |
| stg | 58.33 | 0.00 | 0.00 | 33.33 | 22.92 | 24/24 完成 |
| relief | 50.00 | 8.33 | 20.83 | 5.56 | 21.18 | 24/24 完成 |
| LassoFlexNet 本地文本变体 | 0.00 | 25.00 | 12.50 | 30.56 | 17.01 | 24/24 完成 |
| first_layer_lasso | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 24/24 完成 |
| Localized Lasso | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 24/24 完成 |
| mRMR 原生完整排名 | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 24/24 完成 |
| univariate | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 24/24 完成 |
| deep_lasso | 0.00 | 0.00 | 0.00 | 30.56 | 7.64 | 24/24 完成 |
| lasso | 0.00 | 0.00 | 0.00 | 27.78 | 6.94 | 24/24 完成 |
| cae_keras | 0.00 | 0.00 | 4.17 | 11.11 | 3.82 | 24/24 完成 |
| ShapleyValueSampling | 8.33 | 0.00 | 0.00 | 5.56 | 3.47 | 24/24 完成 |
| adaptive_group_lasso | 0.00 | 0.00 | 0.00 | 13.89 | 3.47 | 24/24 完成 |
| attention_maps | 0.00 | 0.00 | 4.17 | 8.33 | 3.13 | 24/24 完成 |
| canceloutsigmoid | 0.00 | 0.00 | 4.17 | 8.33 | 3.13 | 24/24 完成 |
| canceloutsoftmax | 0.00 | 0.00 | 0.00 | 8.33 | 2.08 | 24/24 完成 |
| Deconvolution | 0.00 | 0.00 | 0.00 | 5.56 | 1.39 | 24/24 完成 |
| DeepLift | 0.00 | 0.00 | 0.00 | 5.56 | 1.39 | 24/24 完成 |
| GuidedBackprop | 0.00 | 0.00 | 0.00 | 5.56 | 1.39 | 24/24 完成 |
| Saliency | 0.00 | 0.00 | 0.00 | 5.56 | 1.39 | 24/24 完成 |
| SmoothGrad | 0.00 | 0.00 | 0.00 | 5.56 | 1.39 | 24/24 完成 |
| FeaturePermutation | 0.00 | 0.00 | 4.17 | 0.00 | 1.04 | 24/24 完成 |
| 随机排名控制 | 0.00 | 0.00 | 4.17 | 0.00 | 1.04 | 24/24 完成 |
| FeatureAblation | 0.00 | 0.00 | 0.00 | 2.78 | 0.69 | 24/24 完成 |
| IG_noMul | 0.00 | 0.00 | 0.00 | 2.78 | 0.69 | 24/24 完成 |
| InputXGradient | 0.00 | 0.00 | 0.00 | 2.78 | 0.69 | 24/24 完成 |
| deeppink | 0.00 | 0.00 | 0.00 | 2.78 | 0.69 | 24/24 完成 |
| fsnet | 0.00 | 0.00 | 0.00 | 2.78 | 0.69 | 24/24 完成 |
| xgboost | 0.00 | 0.00 | 0.00 | 2.78 | 0.69 | 24/24 完成 |
| e2efs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 24/24 完成 |

## 原实验仓库到底是什么基线

固定来源为 [feature-selection@555ba11（2026-07-10）](https://github.com/DreamSoul-AI/Deep-Regularization/tree/555ba11e8872bd6618cd721bd3b08f7128db9e6b)。main/dev根模型与研究分支不同，9月模型以证据快照保留；未发现10月9日或11月A—L的新源码分支。

- 原公开512默认：Mish32×32、500轮/120 warm-up、C=.05、lr=.005、初始ρ200、batch64、WD=.001、feature dropout=.6。维数分段ρ见 [原包装器](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L1338)。本轮新增该行，直接执行固定Git对象内的原模型/训练函数
- 先前公开快照三条线是固定128维配方扩到512，初始ρ50；旧结果保留，不能冒充原512默认
- 10月有效gate参照：共享线性/平方父gate、ReLU32×32、120轮、全局C=.003172、ρ=.18857、lr=.002084；canonical prox，零warm/WD/dropout。按报告恢复，未取得历史原驱动
- 我们的一阶和一阶＋Fisher保持原已实现方法：在raw gate路径上更新score scaling，最终仍按训练后gate排名；本轮没有移植到有效gate骨干或改近端目标

ρ50和ρ200本轮四任务的恢复率恰好一致，但不是重复结果：24折中只有13折的Top-k集合相同，分数向量全部不同，见 [逐折核对](ranking-dashboard/512/rho-reference-comparison.json)。

原清单含RF、TreeSHAP、Relief、FSNet、E2E-FS、mRMR、MI、LassoNet、CAE、两种CancelOut、DeepPINK及10种归因；普通NN是归因骨干。STG、随机控制、XGBoost、Localized Lasso、LassoFlexNet及论文补充选择器也纳入本轮。LassoFlexNet是明确标记的本地文本可行变体，官方代码未取得，不另造官方成绩。

## 原commit保存的历史512维结果

下表来自原commit携带的均值文件，仅作历史背景；缺少执行版本/环境和逐折分数，不能混入本轮配对成绩。

| 历史方法 | XOR | Ring | Ring+XOR | Ring+XOR+Sum | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF | 16.67 | 100 | 75 | 63.89 | 63.89 |
| mRMR | 0 | 100 | 70.83 | 55.56 | 56.60 |
| TreeSHAP | 0 | 100 | 58.33 | 50 | 52.08 |
| LassoNet | 100 | 0 | 33.33 | 47.22 | 45.14 |
| MI | 0 | 100 | 33.33 | 13.89 | 36.81 |
| Relief | 16.67 | 0 | 0 | 0 | 4.17 |

来源例：[LassoNet XOR原始结果文件](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/results/lassonet-xor-1000.txt)。历史mRMR Ring100%与本轮Pearson估计器不同，执行来源尚未闭合，不能直接说方法退化。

## 比较边界与当前证据

本次未显式绑定的旧LassoNet记录，原生构造器实际自动使用CUDA；旧日志的CPU是请求值，Windows上的空CUDA可见性未禁用GPU。当前汇总已更正设备元数据，原分数不改，[设备审计](ranking-dashboard/512/lassonet-device-audit.json)保留依据。未来入口显式传入设备。

Attention Maps的512维物理batch512曾占用约26GB内存，已保留停止记录并改为GPU上16样本分块累积：有效batch仍为512，每批仍只更新一次，模型/学习率不变。实际上游FT模型在关闭dropout时的完整/分块梯度一致；dropout开启会改变随机数次序，不声称逐位同轨迹。Localized Lasso仅按不同任务/折分片并行，求解目标和数值门限不变。

本轮baseline统一训练折z-score；原benchmark先2*x-1、各wrapper自行预处理。Relief保留原生符号；原benchmark统一取abs。mRMR用原生完整贪心顺序，原k集合包装器内部随机分数只留作诊断。LassoNet排名来自路径 feature_importances_，不是最低val_loss checkpoint的β；best-val模型只用于预测。见 [原LassoNet调用](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/core.py#L175)。

所有方法复用相同保存的数据/划分/支持真值。六折训练集重叠且来自一个生成实例，不能冒充多个独立数据生成种子。不同结构与预算是方法间比较，不能作为只换评分的消融。真实相关坐标不参与训练或选正则参数；评价使用真值k，FSNet、CAE和E2E-FS等原生预算选择器也预先给定相同k。128维和512维并非只追加噪声列的嵌套数据，跨维数差异不能全部归因于新增噪声。

[逐折分数与机器汇总](ranking-dashboard/512/summary.json)、[原始分数](ranking-dashboard/512/verified_results.jsonl)、[数据身份](ranking-dashboard/512/cases.json)、[失败记录](ranking-dashboard/512/failures.json)。标准库核验：`python experiments/verify_baseline_evidence.py docs/ranking-dashboard/512`。刷新看板：`python experiments/update_ranking_dashboard.py`。

128维已完成表另留在 [既有证据](<研究记录 Research Record/20261128-baseline-repairs/128/table.md>)，不与512维成绩合并。方法推导与旧运行细节从 [研究计划](<研究计划 Research Plan Deep Regularization.md>) 查阅。
