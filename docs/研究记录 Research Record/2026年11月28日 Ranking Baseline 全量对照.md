# Ranking Baseline 全量对照

本页保留首次816条记录的历史状态。后续已修复 mRMR、完成 LassoFlexNet 本地实现并扩展512维 XOR/Ring，更新的36方法总表、版本核对与证据见 [Baseline 修复与512维复核](<2026年11月28日 Baseline 修复与512维复核.md>)。本页“原方案”均限定为公开 raw-gate 快照500/120轮，不能等同于10月有效 gate GRN-FS 或11月 A—L。

本轮按照新增授权，在候选一已经使用的同一批数据上补跑全部既有 ranking baseline，并接入 Localized Lasso 和 LassoFlexNet 论文中的其他特征筛选方法。范围为四个合成任务、128维、每任务六折；不包含真实数据下游重训练、其他维度网格或候选二至四。

## 一、比较范围与解释边界

这是一轮固定设置的配对 ranking 比较，不是各论文完整的调参复现。所有方法使用同一批保存的数据、特征置换、训练索引和相关特征真值。主指标为真实支持大小 k 下的 Recall@k，另报完整排名 AP 和支持 AUROC。四任务的 k 分别为2、2、4、6；Macro 为四任务均值，不按 k 加权。方法只在训练部分拟合，归因读出使用训练集内固定128行，不用测试标签选参数或计算 importance。

每个任务来自 seed0 的同一份1000样本数据，六折的训练集相互重叠，不能称为六次独立数据生成或六个独立随机种子的确认实验。跨数据生成种子的稳健性仍待下一阶段检验。

本轮沿用各旧 wrapper 的模型和主要训练设置，新增方法采用下文固定参数，未为任何方法用支持真值寻优。不同方法的训练预算、模型结构、内部选模方式不同。因此表中结果能够说明本轮设置的表现，不能推出一个方法在充分调参后一定优于另一个方法，也不能把本表当成 LassoFlexNet 论文的真实数据主表。

我们的三条方法线来自已独立核验的 [原方案复现](<2026年11月28日 原方案复现与候选一 Ranking 对照.md>)。分别为原 column-norm scaling、只用一阶信息、一阶加 empirical Fisher 二阶信息驱动 ADMM；最终都按训练后的 gate 读出排名。它们不是同一个冻结模型上的三种事后归因。

## 二、LassoFlexNet 论文补充的比较对象

[ICML 2026 官方论文](https://proceedings.mlr.press/v306/lui26a.html) 附录的特征筛选表包含 Univariate、Lasso、1L Lasso、Adaptive Group Lasso、LassoNet、Attention Maps、Random Forest、XGBoost、Deep Lasso。这些是本轮需要覆盖的方法，不能只补 LassoFlexNet 一行。

RF 与 LassoNet 已在原比较清单内。本轮新增或显式接入 Univariate、L1 Logistic、First-Layer Lasso、AGL、Attention Maps、XGBoost、Deep Lasso。分类任务中的 Lasso 按上游代码使用 L1 Logistic，1L Lasso 指多层 MLP 第一层的列分组惩罚，二者分开。

论文中的 No FS + MLP/FT 是下游训练参照，TabM、TabPFN、CatBoost 等属于整模型预测比较，不能自动算成 feature ranking 方法。本轮没有因此启动下游训练。

截至本轮核验，[LassoFlexNet 官方仓库](https://github.com/BorealisAI/lassoflexnet) 只有 README，明确写有 code coming soon，没有可运行代码。该方法保留在比较清单中，状态为未取得官方实现；不借用其他方法的结果、不把自行猜写的算法当成官方复现。研究计划中已记录的印刷算法2一致性问题仍待核对。

## 三、完整结果

已完成30种外部 baseline、1条随机排名控制和3条本方案方法线，每种均为24个方法/任务/折记录，共816个成功记录。mRMR、LassoFlexNet 另列状态，不计入完成数。表中 Recall 与 AP 均为百分数。

| Method | XOR | Ring | Ring+XOR | Ring+XOR+Sum | Macro | AP | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原方案 column-norm | 100.00 | 75.00 | 58.33 | 66.67 | 75.00 | 77.81 | 24/24 |
| 候选一：一阶 | 100.00 | 66.67 | 54.17 | 66.67 | 71.88 | 73.85 | 24/24 |
| 候选一：一阶＋Fisher | 100.00 | 66.67 | 54.17 | 66.67 | 71.88 | 72.90 | 24/24 |
| Random Forest | 0.00 | 100.00 | 87.50 | 86.11 | 68.40 | 77.34 | 24/24 |
| treeshap | 0.00 | 100.00 | 66.67 | 77.78 | 61.11 | 65.26 | 24/24 |
| xgboost | 33.33 | 91.67 | 50.00 | 61.11 | 59.03 | 59.87 | 24/24 |
| LassoNet | 100.00 | 0.00 | 50.00 | 66.67 | 54.17 | 55.82 | 24/24 |
| relief | 100.00 | 0.00 | 33.33 | 30.56 | 40.97 | 40.68 | 24/24 |
| mi | 0.00 | 100.00 | 29.17 | 22.22 | 37.85 | 39.51 | 24/24 |
| stg | 91.67 | 0.00 | 16.67 | 19.44 | 31.94 | 31.09 | 24/24 |
| attention_maps | 0.00 | 16.67 | 29.17 | 50.00 | 23.96 | 27.95 | 24/24 |
| Deconvolution | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 12.28 | 24/24 |
| DeepLift | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 12.28 | 24/24 |
| FeatureAblation | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 12.07 | 24/24 |
| GuidedBackprop | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 12.28 | 24/24 |
| InputXGradient | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 11.81 | 24/24 |
| Saliency | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 12.28 | 24/24 |
| SmoothGrad | 8.33 | 0.00 | 0.00 | 30.56 | 9.72 | 12.23 | 24/24 |
| first_layer_lasso | 0.00 | 0.00 | 0.00 | 38.89 | 9.72 | 13.11 | 24/24 |
| canceloutsigmoid | 8.33 | 0.00 | 8.33 | 16.67 | 8.33 | 11.22 | 24/24 |
| deep_lasso | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 12.52 | 24/24 |
| lasso | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 12.73 | 24/24 |
| Localized Lasso | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 11.78 | 24/24 |
| univariate | 0.00 | 0.00 | 0.00 | 33.33 | 8.33 | 11.49 | 24/24 |
| canceloutsoftmax | 8.33 | 0.00 | 12.50 | 11.11 | 7.99 | 10.76 | 24/24 |
| FeaturePermutation | 0.00 | 0.00 | 0.00 | 30.56 | 7.64 | 11.36 | 24/24 |
| adaptive_group_lasso | 0.00 | 0.00 | 0.00 | 30.56 | 7.64 | 12.51 | 24/24 |
| IG_noMul | 0.00 | 0.00 | 0.00 | 25.00 | 6.25 | 10.24 | 24/24 |
| 随机排名控制 | 8.33 | 8.33 | 8.33 | 0.00 | 6.25 | 9.02 | 24/24 |
| ShapleyValueSampling | 0.00 | 0.00 | 0.00 | 25.00 | 6.25 | 9.76 | 24/24 |
| cae_keras | 0.00 | 0.00 | 0.00 | 11.11 | 2.78 | 3.65 | 24/24 |
| e2efs | 0.00 | 0.00 | 4.17 | 2.78 | 1.74 | 3.05 | 24/24 |
| fsnet | 0.00 | 0.00 | 4.17 | 2.78 | 1.74 | 4.64 | 24/24 |
| deeppink | 0.00 | 0.00 | 0.00 | 5.56 | 1.39 | 5.02 | 24/24 |
| mRMR | — | — | — | — | — | — | 编译依赖缺失 |
| LassoFlexNet | — | — | — | — | — | — | 官方代码未发布 |

原 column-norm 在本轮 Macro Recall@k 最高；候选一的一阶及一阶加Fisher均未超过原方案。随机森林在 Ring 和两个混合任务优于原方案，但 XOR 没有恢复真支持，因而总均值低于原方案。LassoNet 在 XOR 全恢复，Ring 为零；这同样只反映本轮旧 wrapper 的30轮初始化和固定路径设置。Localized Lasso 在三个非线性任务没有恢复真支持，含Sum任务恢复约三分之一。不能据此宣称充分调参后的方法排序相同。

## 四、实现与固定参数

旧方法取自本仓库固定提交 [555ba11](https://github.com/DreamSoul-AI/Deep-Regularization/tree/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src)，通过临时 staging 运行，不把历史快照当成当前主模块，也没有 checkout 或改写旧分支。共有十种 Captum 归因方法，以及 RF、MI、Relief、FSNet、STG、LassoNet、CAE、CancelOut 两种版本、DeepPINK、TreeSHAP、E2E-FS、mRMR。完整参数和软件版本以证据包 manifest 为准。

| 方法 | 本轮设置与读出 |
| --- | --- |
| RF / TreeSHAP | 500棵树、原固定随机种子，CPU并行数限制为2；RF使用 impurity importance，TreeSHAP 对固定128个训练样本的绝对归因取均值 |
| MI / Relief | 使用固定旧代码的默认实现，Relief 保留有符号 relevance |
| 随机排名控制 | 原 random 分支；每折重置同一 seed=0xCAFE，生成128维均匀随机分数并独立登记，不拟合模型 |
| 十种 Captum 方法 | 同一折共用一个拟合的旧 NN；五层58宽 Mish 网络、原噪声及 dropout、Adagrad，最多416轮与原 early stopping。保留原 target=0 和各 wrapper 的归因聚合 |
| CancelOut / DeepPINK | 原 NN wrapper 与训练默认值；DeepPINK 使用同一个原生成器、数据 seed 和置换重建的真实 synthetic knockoff，不填零矩阵 |
| FSNet | 原 k 模型、2000轮默认训练；不额外拟合仅用于另一套2k预测指标的独立模型 |
| STG | 官方包0.1.2，32×32 ReLU，300轮，lambda=.1、sigma=.5、Adam lr=.001，概率 gate 读出 |
| LassoNet | 官方包0.0.22，32×32，初始化/路径各最多30轮，batch64、patience10、lambda_start=3.2768、tol=.9999；使用原生路径 feature_importances_ |
| CAE | concrete-autoencoder 0.0.1 原 Keras selector，独立 tf-keras 环境；原监督 decoder，300轮、温度10到.01、lr=.0001、tryout_limit=1，读取 native mask |
| E2E-FS | 固定提交附带的 PyTorch 实现，three_layer_nn、batch64、50轮、内部20%验证，读取 native mask；不安装同名但不同实现的 PyPI 包 |
| Univariate / Lasso | 上游 SelectKBest 对应的 ANOVA F score；L1 Logistic 使用 saga、C=1、seed1，max_iter=10000 防止提前触发迭代上限 |
| First-Layer Lasso / Deep Lasso | 固定上游100×100 ReLU MLP、dropout=.1；AdamW lr=.001、weight decay=.0002，最多200轮，batch512，40/80轮衰减、5轮线性 warmup；reg_weight=.1 |
| AGL | 在相同 MLP 上先做第一层 Group Lasso，再以初始列范数的平方倒数作固定权重训练第二阶段；gamma=2、epsilon=1e-8、reg_weight=.1。属于显式标注的本地两阶段实现 |
| Attention Maps | 固定上游 FT：192维 token、3层、8 heads，保留上游 dropout；训练设置同新增 MLP，去掉正则；按128个训练样本、各层与各 head 的 CLS-to-feature attention 均值读出 |
| XGBoost | 100棵树、depth6、lr=.3、固定 seed；使用原生 feature_importances_，未运行论文的 Optuna 搜索 |

新增 MLP/FT 训练适配器每20轮根据内部验证 BCE 保存最优参数，一次检查无改善即停止；这与上游完整管线按验证预测指标逐轮判断的实现不同。Deep Lasso 保留上游对 batch-mean task loss 求输入梯度、再跨样本取 L2 的正则公式，因此其强度依赖 batch。AGL、正则系数和验证设置均为预先固定的迁移设置，不冒充作者选择的最优参数。

十种归因方法不是十个独立训练模型。旧骨干为 Mish，部分只为 ReLU 设计的 Captum 变换不会提供预期的独立传播规则，表中若出现相同结果应按这一实现条件解释。

E2E-FS 的原50轮 wrapper 预算只有约500次训练更新，低于其 soft-mask 默认2000步 warmup；CAE/E2E-FS 等还可能只给离散 mask，存在大量平局。这些结果反映本轮原设置，不能用低分宣称该类方法本身无效。原生 mask 不提供完整连续 ranking 时保留原值，平局按列索引稳定排序，不另用真值打破平局。

额外只读检查了已保存的24个归因骨干，没有再训练或据此调参。XOR、Ring、Ring+XOR、Ring+XOR+Sum 的平均训练预测 AUROC 分别约.773、.813、.746、.846，外层保留集约.519、.499、.486、.542。旧 NN 设置没有学到可靠的外层预测规律，因此这些归因方法的低 ranking 分数不能单独归因于归因规则本身；后续需要按事先约定的预测验证协议校准骨干。这里是冻结骨干的质量诊断，不是选后特征下游重训练。

### Localized Lasso 的适配与数值核验

使用作者 [Localizedlasso 固定版本](https://github.com/myamada0321/Localizedlasso/tree/f7944f2913696a3fd0fb67d4e43552af69797d52)，仅先转换 Python 2 print 语法。以0/1标签做原平方损失回归，使用训练特征上的5近邻图并取 `(R+R.T)/2`，不使用测试图。保留默认10次 IRLS、lambda_net=1、lambda_exc=.01、无 bias。每个样本有一个系数向量，最终对同一 feature 的系数跨训练样本取 L2 范数，形成全局 ranking；这与在已有 MLP 前接 gate 不同。

第一个完整折直接运行作者求解器。其大规模 `D^-1 A.T` 中间量很耗内存和时间，后续采用按 feature 分块的 Woodbury 计算，以带残差检查的 CG 解同一个样本空间线性方程，保留初始化、IRLS更新和所有 epsilon。没有降低样本量、特征数或迭代次数。

独立小矩阵测试将该求解器与直接解完整稠密正规方程比较。完整首折再与作者实现比较：最大系数差约6.28×10^-7，系数整体相对 L2 误差约5.70×10^-7，最大 importance 差约8.99×10^-7；完整 feature 排名 Spearman=1、Top-k 相同。时间从约400.5秒降为43.2秒。它是有明确数值容差的等目标求解适配，不宣称逐浮点位一致；较严格的早期误差门限未通过的记录也保留在本地。

### 兼容修复与未运行项目

- PyTorch 新版删除 ReduceLROnPlateau.verbose，临时源码移除该关键字。
- 旧 NN 验证 checkpoint 原来保存实时 tensor 引用，已改为 clone。这是 correctness 修复，因此本轮 NN 结果不冒充逐字节恢复历史报表。
- STG 的 collections.Sequence/Mapping/Set 改用 collections.abc 对应接口；E2E-FS 修正 staging 后的 vendor 相对目录。
- SHAP 新版数组布局显式选择分类的 class1；CAE 使用真实 Keras 原实现，不用仓库内 PyTorch proxy 替代。
- 汇总复核发现 Relief 的负 relevance 不能统一取绝对值，已从保存的24个原始有符号向量重算指标，并增加负值与平局的独立测试。早期错误读出未进入最终证据表。
- mRMR 尝试安装原 pymrmre；修复本机 torch/numpy 原生 DLL 加载顺序后，安装继续失败于缺少 Microsoft Visual C++ 14+ 编译工具。当前 WSL 没有已安装发行版、Docker daemon 未运行，未启动新系统服务或安装系统级构建工具。该项没有实验结果，旧 mRMR 历史数字未回填。
- LassoFlexNet 尚无官方实现，原因见第二节。

## 五、核验与复跑入口

[便携证据包与复跑命令](20261128-baselines-128/README.md) 包含完整逐折结果、参数、来源与失败登记；不依赖训练环境的独立指标核验已通过816条分数向量及34条完整方法线。

数据文件 SHA-256、来源提交、适配补丁、实际软件版本、逐折分数、原始失败与核验结果随证据包保存。旧的不同数据运行和中断 CPU 试跑单独留档，不进入完整表。

准备环境时先沿用项目现有 PyTorch，再在项目临时目录创建隔离环境。通用增补依赖见 [requirements-baselines.txt](../../experiments/requirements-baselines.txt)，CAE 按该文件说明单建环境。源码准备使用 [prepare_baseline_sources.py](../../experiments/prepare_baseline_sources.py)，目标须为空目录，按固定提交提取/下载并记录 hash。

完整队列入口为 [run_baselines.py](../../experiments/run_baselines.py)，逐方法入口为 [baseline_suite.py](../../experiments/baseline_suite.py)、[baseline_supplement.py](../../experiments/baseline_supplement.py)、[baseline_neural.py](../../experiments/baseline_neural.py)、[baseline_localized.py](../../experiments/baseline_localized.py)。支持逐方法、逐折区间恢复；并行 worker 使用各自输出目录，成功记录按 method/task/fold 去重，失败重试保留。

[summarize_baselines.py](../../experiments/summarize_baselines.py) 从每折128个分数重新计算 Recall@k、AP、AUROC 和 Top-k，拒绝有冲突的成功重试，不把失败算成零分或把未完成任务做成完整 Macro。它还记录常量 score 情况。此次线性代数适配与原有 score/原始训练一致性测试均已通过。

本轮只更新现有实验分支与 PR。下游任务、其他候选和默认算法没有因此获得结论。
