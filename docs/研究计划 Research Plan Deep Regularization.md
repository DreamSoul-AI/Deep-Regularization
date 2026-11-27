# 研究计划 Research Plan Deep Regularization

## 一、研究目标与实验分工

研究问题是：在已有网络上接入输入 gate，以 PQI / ADMM 形成重要性分数，能否改善 feature ranking，以及这些排名用于特征选择后能否带来更好的下游预测效果。保留四个方法候选，通过两块正式实验分别回答这两个问题。

| **实验块** | **数据与任务** | **主要评价** | **保留个数的处理** |
|---|---|---|---|
| A：Feature ranking | 原有 Feature-Selection-Benchmark 的 XOR、Ring 与混合任务，已知真实支持集 | recall@k 为主，补充完整排名 AP / 恢复 AUROC、稳定性和成本 | 方法输出分数，评测端用真值 k 计算命中，不要求方法自动发现 k |
| B：下游预测 | 按 LassoFlexNet §7.1 及其引用的十二真实数据集 benchmark | 选定特征后训练统一的 MLP / FT-Transformer，分类 accuracy、回归 RMSE | 主比较按原协议固定 k；正则参数按验证集选，自动选 k 另列扩展 |

两块分别给结果表和结论。真实支持恢复、对指定旧模型的忠实性、重训后的预测价值不互相替代。TabZilla 与 TabReD 是 B 的后续扩展，不增加第三块主任务，也不替代 A 的相关特征真值。

本计划整理日期为2026年11月27日。已完成文献定向阅读、已有代码及历史结果定位、数据和选模协议核对；尚未实现四候选的新配置或运行本计划实验。科学贡献与性能收益均为待检验主张，执行前还需冻结第九章列出的参数和接口。

## 二、方法候选与待检验问题

### （一）共同约定

原始数据 X 固定。前三个候选保持输入乘法 gate 与原网络的组合，第四个单独采用 LassoNet 结构。分类时 gate 控制一个原始输入 feature，对全部类别共享，不因输出类别数增加而变成多个独立输入选择器。

score 由预测任务损失产生，作为稀疏更新的 scaling 时执行 stop-gradient，不新增一个可训练 score，也不优化 PQI(score)。稀疏项、辅助变量坐标、importance 读出分开登记。用 g 表示共享 gate，G 表示按稳定 sample ID 保存的逐样本 gate，Z 为稀疏辅助变量，U 为缩放对偶量。

对当前候选的稀疏坐标 A(θ)，ADMM 模型块优化任务损失加一致性项，辅助块更新稀疏副本，再更新对偶量：

$$\theta^{k+1}\approx\arg\min_\theta\left\{\mathcal L(\theta)+\frac\rho2\lVert A(\theta)-Z^k+U^k\rVert_F^2\right\}.$$

$$Z^{k+1}\in\arg\min_Z\left\{\lambda R_{c^k}(Z)+\frac\rho2\lVert Z-A(\theta^{k+1})-U^k\rVert_F^2\right\}.$$

$$U^{k+1}=U^k+A(\theta^{k+1})-Z^{k+1}.$$

向量时使用相应 L2 范数。每轮刷新 c 时属于迭代重加权，有限步神经网络更新也是近似子问题，不据此声称固定目标 ADMM 的全局收敛。λ 表示惩罚强度的通用记号；映射到现有代码时记录实际 C 等配置名，ρ 不能当成同一个参数。

### （二）候选一：只改 score

保持所选旧版本的网络、gate、稀疏坐标和求解器，仅把旧 scaling 的评分来源替换为一阶加经验 Fisher 曲率代理。对每条样本的任务损失 ℓ_n，定义：

$$h_n=x_n\odot g,\qquad t_{nj}=g_j\frac{\partial\ell_n}{\partial g_j}=h_{nj}\frac{\partial\ell_n}{\partial h_{nj}},\qquad s_{nj}=-t_{nj}+\frac12t_{nj}^2.$$

平方先于跨样本平均，保留原始有符号 s。最终排名取绝对值的位置、非负 importance 到 scaling 的映射和刷新频率执行前固定。逐样本梯度平方平均不等于 batch 平均梯度的平方；评分不包含 ADMM 增广项梯度。评分阶段固定 dropout，处理 BatchNorm 的跨样本耦合与损失均值缩放。

检验问题是：与同协议的一阶评分相比，增加平方项是否改善 A 的排名恢复，并在 B 中改善相同保留预算的预测结果。若收益仅来自更多计算、另换骨干或只在调参数据出现，不支持评分本身更有效的结论。经验 Fisher 是曲率代理，逐样本梯度全退化时它也会退化。

### （三）候选二：有效幅度与联合剪枝

普通前向仍为 W(g⊙x)，不在 forward 归一化权重。定义带符号的一致性坐标 q 和非负幅度 a：

$$q_j=g_j\lVert W_{:,j}^{(1)}\rVert_2,\qquad a_j=|q_j|,\qquad A(\theta)=q(g,W).$$

模型块的一致性项通过 q 同时向 gate 和第一层权重反传。列范数是权重的派生量，不另外独立训练。按同一 score 比较原 gate 与有效幅度坐标，再单列与候选一的组合，不能把组合收益只归于一个改动。

这种坐标对 gate 与紧邻权重列的反向缩放保持不变，检验其是否减少尺度补偿引起的读出歧义。它在保留研究快照中已有相近实现，不能包装为完全新概念。最终确定永久删除某输入时，同步清零 gate、整列权重及相应优化器状态。A 只读排名时，不以辅助 Z 为零强迫新增硬剪枝；Z 稀疏和实际预测路径删除分别记录。

### （四）候选三：独立 per-sample gate

保存 G∈R^{N×d}，前向为 f_W(x_n⊙G_n)，W 共享，G 的各行独立。ADMM 跟踪 Z=G，Z/U 与 G 同形。PQI 按整张矩阵的明确向量化定义处理，不能以每个 batch 的独立比值代替同一全局目标。每条样本保存 s_nj，再按固定的绝对值和聚合规则形成全局排名。

主要问题是局部 importance 与聚合排名是否更有用；不以新样本预测泛化作为该候选唯一验收条件。A 的现有任务先评价全局排名，局部真值数据是进一步机制验证，不把未构造的数据集算作已有 benchmark。

B 中仅在训练样本上拟合 G 或 Localized Lasso 的局部系数，聚合训练样本的重要性得到全局排名，再取同一预算的原始列交给统一下游模型。这样不需要为测试样本用测试标签拟合 gate。若另做逐样本预测，必须单独定义未见样本的参数生成规则，不能沿用这一结果冒充部署预测。

局部 G_nj 为零不等于整个 feature 已全局删除，不能因此清零共享权重列。sample ID、G/Z/U、优化器和检查点一起持久化，正确处理全局增广项与 mini-batch 的尺度。存储和逐样本梯度成本进入比较。

### （五）候选四：ADMM-LassoNet，再替换 PQI

先保持 LassoNet 的线性 skip 和 hierarchy，仅将原 Lasso 目标转为 ADMM；核对通过后才替换 PQI。令 B 为 skip 矩阵，W¹ 为第一层，原目标为：

$$\min_{B,W}\ \mathcal L(Bx+f_W(x),y)+\lambda\sum_j\lVert B_{:,j}\rVert_2\quad\mathrm{s.t.}\quad\lVert W_{:,j}^{(1)}\rVert_\infty\le M\lVert B_{:,j}\rVert_2.$$

采用稀疏副本 Z=B 与 hierarchy 副本 P=B、T=W¹。模型块包含三项一致性二次项；第一阶段对 Z 用列组 L1 近端，对 (P,T) 做联合 hierarchy 投影，再更新各自对偶量。实际模型的约束违反量与副本残差同时检查，不把约束副本可行当成实际模型每步可行。

第二阶段仅替换 Z 的稀疏项。例如以 b_j=‖Z_:,j‖₂ 为输入的比值为 (Σ_j b_j)/√(Σ_j b_j²)，但实际 PQI 的指数、归一化和零点定义仍需固定。该比值耦合不同 feature，普通 group soft-thresholding 不再是它的完整精确解。ADMM 分裂不自动提供正确的 PQI 求解器。

比较原版 LassoNet、ADMM-Lasso、ADMM-PQI，分别检验求解器变化与惩罚变化。暂不把前三个候选的 score scaling 混进本候选基本版。其重要性沿用原生路径退出强度或事前固定的对应读出，不与最终预测 checkpoint 的参数混为一谈。

## 三、实验 A：原 Benchmark 的 Feature Ranking

### （一）数据与协议

采用固定的 [feature-selection 分支 benchmark 入口](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/main-benchmark.py) 及对应生成器。主任务如下，N=1000，沿原入口6折评价并固定数据、特征置换和划分。

| **任务** | **真实相关特征数 k** | **原入口的输入维度 d** |
|---|---:|---|
| XOR | 2 | 2、4、8、16、32、64、128、256、512、1024、2048 |
| Ring | 2 | 8、16、32、64、128、256、512、1024、2048 |
| Ring+XOR | 4 | 4、8、16、32、64、128、256、512、1024、2048 |
| Ring+XOR+Sum | 6 | 6、8、16、32、64、128、256、512、1024、2048 |

先用128维完成小规模接口与读出检查，再按上表形成完整维度结果。250/500样本等历史设置不自动混入N=1000主表。DAG 沿原入口保留为单列扩展：原调用含2000维，但 loader 会返回实际 k/k2，数据文件、两个支持定义和样本数未核对前不凭调用字面写成固定真值6。

每次数据生成保留真值、置换与划分。相关特征的位置先置换，避免按原始列顺序或平局规则获得有利结果。训练所需预处理只在允许的训练数据拟合，若为历史精确复核保留不同预处理，单独标记，不把两种协议混用。历史21个外部方法的结果与后续 Base/Five 不是完整配对，本计划新比较必须使用同批数据。

### （二）读出、指标和参数选择

各方法提交长度 d 的原生重要性分数；归因方法另保留逐样本分数。主指标为 recall@k=前k名命中的真实特征数/k。k 只在评测端使用，不要求我们的方法产生恰好k个非零 gate。FSNet、CAE 等原方法本身需要选择数时，保留其输入依赖并标注“给定k”的对照，不能称为未知预算方法。

保留既有 best-2k 指标以便追溯，注明其分母仍是真特征数k；补充完整排名 AP、恢复 AUROC 与支持/排名稳定性。全维度或某任务缺少负类时，AUROC 等不适用指标记为 N/A，不强算。大量零或并列分数采用预先固定、与真值位置无关的规则，并报告退化情况。

A 的参数先采用可核实的原版配置作为参照。新增参数若需调优，只在独立开发数据实例上用同一排名评价和同等搜索预算选择；选定后在独立确认实例上固定运行，不用最终实例真值选λ、checkpoint、绝对值方式或刷新频率。已知支持只用于这个明确标记的开发评价与最终评分，不当作每个待评价样本的输入。

报告各任务、维度的平均及波动，跨四任务采用等权宏平均。区分数据生成 seed、模型初始化和折，不能将同一实例的多个初始化视作完全独立证据。主表不要求训练下游模型来定义排名好坏。

## 四、实验 B：按 LassoFlexNet 协议评价下游效果

### （一）主数据集与额外特征

承接 [NeurIPS 2023 十二数据集特征选择 benchmark](https://github.com/vcherepanova/tabular-feature-selection) 与 LassoFlexNet §7.1、C.2–C.3 的对应设置。以下为扩展前的数值特征维度和原数据规模，实际样本与预处理以所固定数据包为准。

| **数据集** | **配置名** | **样本数** | **原维度** | **任务** |
|---|---|---:|---:|---|
| ALOI | aloi | 108,000 | 128 | 1000类分类 |
| California Housing | california_housing | 20,640 | 8 | 回归 |
| Covertype | covtype | 581,012 | 54 | 7类分类 |
| Eye Movements | eye | 10,936 | 26 | 3类分类 |
| Gesture Phase Prediction | gesture | 9,873 | 32 | 5类分类 |
| Helena | helena | 65,196 | 27 | 100类分类 |
| Higgs 98k | higgs_small | 98,050 | 28 | 二分类 |
| House 16H | house | 22,784 | 16 | 回归 |
| Jannis | jannis | 83,733 | 54 | 4类分类 |
| Microsoft | microsoft | 1,200,192 | 136 | benchmark 按回归处理 |
| Otto | otto | 61,878 | 93 | 9类分类 |
| Year | year | 515,345 | 90 | 回归 |

Microsoft 原 benchmark 实验限制为500,000样本，不能按原表全量120万宣称训练规模。House 的命名与文件身份按配置 house 及原 Appendix B 核对。数据从上游 [2021数据包入口](https://github.com/Yura52/tabular-dl-revisiting-models#33-data) 和 [2022数据包入口](https://github.com/Yura52/tabular-dl-num-embeddings#data) 获取。它与 A 的项目不是同一个 benchmark。

主对标使用 random_feats 与 secondorder_feats，各取50%和75%额外特征比例，共48个数据设置。上游提供的 corrupted_feats 另列扩展，共增加24个设置。比例p按扩展后的数值列数定义，新增列数为 floor(d·p/(1−p))，50%与75%对应约2倍、4倍总维度。原始列和二阶额外列不能简单解释成“真实相关与完全无关”的划分。

先接 California Housing、House 16H、Eye Movements、Gesture、Higgs 98k、Jannis 六个数据集检验回归、二分类、多分类接口，随后补齐12个主表。先行子集不代替完整对标。数据划分采用原协议0.65/0.15/0.20，落实具体 seed、文件与样本索引，不以同名数据的另一个下载版本替换。

### （二）固定预算的主比较

为与该特征选择 benchmark 对齐，令 k=d_original，即加入额外特征前的原始列数。每个选择器提交训练数据产生的全局排名，从扩展后的全部输入列中取前k列，不能预先排除新增列或只在原始列中排序。选择单位是输入feature而非embedding坐标。随后分别训练相同协议的 MLP 与 FT-Transformer。最终报告独立测试集上的分类 accuracy、回归 RMSE，按数据集与下游骨干分表。Top-k 只是统一选择预算，不主张算法自动发现了最优个数。

覆盖范围也按原表对齐：MLP覆盖50%和75%，FT-Transformer的原主表重点覆盖50%。不能把48个数据设置默认都乘上两个骨干，再称为原论文完整复现。FT-Transformer的75%设置若新增，另标本项目扩展；缺少对应作者结果的格子只报告自行运行值。

设置 No-FS 的同一下游骨干作为参照。下游模型从所选列重新训练，与“在旧模型中置零或删除输入但不训练”分开。候选三和 Localized Lasso 按训练样本聚合得排名，下游验证和测试不拟合带标签的局部参数。归因 baseline 在训练选择器阶段产生排名，不使用最终测试标签挑特征。

另保留 LassoFlexNet 按原生结构与自动稀疏路径训练的完整方法行，以对齐论文的整模型结果。它与“选择器→统一下游”的行分组呈现；如果从 LassoFlexNet 导出排名再训练统一下游，该行明确标记为管线适配，不能称为其论文原生数字。原论文部分外部结果引用已有文献，不能与本项目统一重跑结果不加说明地拼成配对表。

分类与回归的数字不直接平均。跨数据集的总体比较采用各任务内一致方向的平均秩等预先固定方式，保留每数据集明细及失败数。原论文用负RMSE统一展示方向时，可作为展示转换，但最终结果仍能回溯原单位。

### （三）后续扩展数据

TabZilla 使用 [官方36个hard task名单及OpenML ID](https://github.com/naszilla/tabzilla/blob/main/scripts/HARD_DATASETS_BENCHMARK.sh)，按数值/混合类型、任务与可用实现记录筛选。LassoFlexNet 自述35个任务，所读 Table 10 实际有33个命名行，故不自行拼出所谓完全相同的35任务复现。

TabReD 采用 [官方预处理版本和划分](https://github.com/yandex-research/tabred/blob/main/readme.md)。8个数据集为 Homesite Insurance、Ecom Offers、Homecredit Default、Sberbank Housing、Cooking Time、Delivery ETA、Maps Routing、Weather。分别报告随机与时间划分。当前下载入口需Kaggle身份，尚未下载全数据。它用于B的规模及漂移扩展，不提供全列相关性真值。

## 五、Baseline 与实现状态

| **方法组** | **必须保留的对照** | **本次核对状态与接入要求** |
|---|---|---|
| 传统选择 | Random Forest、Mutual Information、Relief、mRMR | A 的既有分派与历史结果可定位；B 使用对应输入适配与相同下游流程 |
| 神经选择 | 原版LassoNet、FSNet、DeepPINK、Concrete Autoencoder | 既有分派存在；保留k输入、knockoff等依赖，记录分类/回归支持范围 |
| 全局门控 | CancelOut sigmoid、CancelOut softmax、STG | CancelOut有历史结果，softmax保持历史secondary身份；STG在根src接入，嵌套benchmark未注册，先补入口验证 |
| 端到端选择 | E2E-FS | 嵌套benchmark有分派及vendor包装器，根core未接同名分支，现有历史主表无完整对应结果 |
| 输出归因 | Saliency、Input×Gradient、IG_noMul、SmoothGrad、GuidedBackprop、DeepLift、Deconvolution、FeatureAblation、FeaturePermutation、ShapleyValueSampling | 10个Captum分支均可定位。先保留原生读出，核对目标输出、类别与样本范围，按mean(abs(attr))聚合；不把target=0写成多分类损失归因 |
| 树归因与控制 | TreeSHAP、普通NN / No-FS、随机排名 | TreeSHAP虽放在FS分派中但属归因；随机排名已在11月28日128维配对比较中接入并核验。普通NN已作为归因骨干拟合和诊断，No-FS下游重训练仍待B阶段 |
| 新增重点 | LassoFlexNet、Localized Lasso | 均纳入正式baseline。前者当前官方仓库仅README且code coming soon；后者方法已核读，原实现、回归/分类适配及局部到全局读出待核验 |
| B的原生对照 | Deep Lasso、First-Layer Lasso、Adaptive Group Lasso、Univariate、XGBoost、RF、Transformer attention | 沿十二数据集上游实现接入，不因A未包含这些方法就省略；Deep Lasso已有输入梯度正则实现 |
| 本研究对照 | 原方案、候选1/2/3、原版LassoNet、ADMM-Lasso、ADMM-PQI及明确标记的必要组合 | 固定版本、读出与调参协议后分别比较，避免同时改多个因素却归因为单项 |

本表是应覆盖的比较矩阵，不代表每个方法已具备所有任务接口。缺少回归、多分类、knockoff或未见样本规则时，记录不适用或待适配格，不以零分冒充运行结果，也不静默删除难以运行的方法。Sequential Attention 的既有记录尚未定位到闭合调用，作为待核条目，不标作当前已可运行基线。

核心参考包括 [LassoNet](https://jmlr.org/papers/v22/20-848.html)、[LassoFlexNet](https://proceedings.mlr.press/v306/lui26a.html)、[Localized Lasso](https://proceedings.mlr.press/v54/yamada17a.html)、[STG](https://proceedings.mlr.press/v119/yamada20a.html)、[Group Fisher](https://proceedings.mlr.press/v139/liu21ab.html)、[Taylor pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)。11篇新增PDF已正式入库并同步飞书，INVASE、Reweighted ADMM未取得PDF，不计作已入库。相关工作与阅读深度见 [四候选相近文献](<调研 Search/文献调研 Literature Search/文献调研 Literature Search Deep Regularization.md>)。

## 六、正则参数与选模规则

### （一）统一规则

固定预算时，各方法使用同一个k。选择器正则、ρ、优化参数和下游模型的配置只用训练/验证信息选择；选定配置后才计算测试指标。相同的是选择准则与可比预算，不要求不同方法采用相同数值的λ或同一范围。测试分数不用于选择checkpoint、路径位置、score公式或保留数。

B主协议沿十二数据集benchmark和LassoFlexNet C.2：以验证集下游accuracy或RMSE等预先固定的任务指标选择配置，以每个“数据设置×选择器×下游骨干”单元100次Optuna搜索为对标预算口径，选定后报告10个初始化seed的测试表现。先行接口检查不冒充完成这个全预算实验，完整批次的总算力另行安排。优先复用上游具体搜索空间、数据与骨干配置，记录实际试验数及早停；若原文对某些大数据模型使用默认参数或缺少相应表格格子，如实保留，不补造成完全统一的作者实验。

原论文LassoFlexNet采用最多200轮预训练、每个λ最多100轮，两个阶段patience为20。其λ起点搜索区间为10⁻⁸至10⁻⁶，终点倍率为10³至10⁵，使用幂间隔路径；这些是作者设置，不能直接当作我们PQI的搜索范围。M、路径长度、EMA和具体骨干配置在获得可核实实现后冻结，不能凭缺失信息补成“官方默认”。TabZilla/TabReD所述40次搜索、ASHA与额外划分是另一协议，不替代B主实验的100次设置。

B主表不让每个方法自行选择不同k。若后续单列自动选预算实验，则所有方法在同一验证目标下选择支持数，并计入逐支持重训成本；该扩展不改变固定k主结果。

### （二）不同方法的正则选择

| **方法** | **需要选择的稀疏参数** | **本计划如何选择** |
|---|---|---|
| 原方案与前三候选 | 代码中的C或等价λ、ρ，以及所选PQI/scaling配置 | A在独立开发实例固定后确认；B按统一验证目标和预算搜索，完整记录映射到代码的名字、范围和实际目标 |
| 原版LassoNet | λ路径、M及网络配置 | 使用原生path或CV接口，保存选中状态与支持。B若重训统一下游，按下游验证结果选配置，不能用LassoNet自身得分代替 |
| ADMM-Lasso / ADMM-PQI | λ、ρ、M与路径/停止配置 | 第一阶段先与原版对齐结构和评价，再替换PQI。不假定比值惩罚的支持单调，有限候选由验证表现选择 |
| LassoFlexNet | 原生稀疏路径与其结构/优化参数 | 复用作者可核实协议，原生整模型与导出排名后的下游适配分开调参和报告 |
| Localized Lasso | λ₁图正则、λ₂平方L1及样本图参数 | 原文有这两个目标参数，不能误读成无须调参。B按训练局部系数聚合后的下游验证表现选择 |
| 其他baseline | 各自原生参数及必要的k | 复用既有或上游官方范围，按A/B各自评价准则与同等搜索预算选择 |

官方 [LassoNet接口](https://lasso-net.github.io/lassonet/api/) 的auto lambda_start只估计路径起点。普通path从稠密模型warm start，逐步增大λ，输出不同支持数的状态，需显式选择并load。ClassifierCV / RegressorCV按各折得分选best_lambda_；当前项目包装器则取path上val_loss最低状态。两者均不是脱离验证数据的最优feature个数保证，也不能将路径跑到末端误当成已选最优模型。

若固定外部预算K，路径未必恰好经过每个整数K。采用排名Top-K或从满足预算的路径点选择时，明确二者不同，不以辅助变量零数冒充已选择K个有效输入。ρ控制一致性，不代替λ的统计正则作用，M控制hierarchy，也不等于预算K。

## 七、有效比较与结果判读

A与B都固定数据身份、划分、随机性和任务定义。报告原始分数、最终排名、选择支持、有效模型路径与完整参数搜索记录。墙钟、峰值显存、模型/梯度求值量、全λ路径和per-sample状态开销一起记账；失败、超时和数值异常保留在覆盖统计中。

两块分别形成：A的任务×维度×方法排名恢复表，B的数据集×额外特征类型×比例×下游模型×方法预测表。主比较配对同实例、同划分和可比随机性，不以旧文献均值替代配对重跑。跨任务宏平均不扩大独立样本数，发现性调参和独立确认分开。

若A改善而B不改善，结论限于排名恢复，不能声称下游获益；若B改善而A不改善，可能是模型或数据条件带来的预测收益，不能反推恢复了真实支持。仅在特定数据/维度有效时按条件报告。排名正确也不保证固定旧模型的删除响应一致，真实支持与模型实际依赖须分别解释。

只做足以解释主要差异的机制检查：同起点比较一阶、纯平方项和两者之和；有效幅度的保持预测缩放；ADMM模型与副本残差；候选四的实际hierarchy违反量。它们服务于两块主结果，不替代数据集与baseline横向对照。不得只因Z稀疏或优化损失下降就宣布完成feature selection。

## 八、执行顺序与交付

1. 冻结代码、数据身份和参数注册表，先校验所有baseline入口、任务输出形状、score、Top-k与平局规则。现有根src与嵌套benchmark不同，不能依赖偶然的Python导入顺序
2. 在A的128维完成接口检查和配对比较，再扩到原维度网格。保留原方案，四候选分别登记，候选四按ADMM-Lasso到PQI顺序推进
3. 接通B的六数据集先行子集，再补齐12数据集与48个主设置。用固定k和统一下游验证选模，保留LassoFlexNet原生整模型行
4. 汇总A与B的独立确认结果、资源与失败，必要时做少量机制诊断。自动预算、corrupted、TabZilla、TabReD只按标记扩展

正式成果包括两块主结果表、逐数据集/任务明细、搜索与选模依据、参数及数据版本、独立确认结果和必要诊断。完整计划的负责人、执行期限及全批总预算尚未分配，不编造承诺。计划初次交付时尚未启动训练；11月27日至28日用户进一步授权候选一及128维 ranking baseline 的代码实现与配对实验，实际状态见下方实施记录，不扩展为全部候选或下游实验已经开展。

## 九、通读复核、实施依赖与资料入口

本轮已通读方法稿与文献/benchmark稿，并统一以下问题：两块均为正式研究内容；固定k与验证选λ分开；排名恢复与下游预测分开；per-sample参数不使用测试标签支持无标签预测；真实数据原始列不冒充相关性真值；保留既有全部基线而非只列新增两篇。

执行前必须冻结的技术项如下，它们是研究实现依赖，不冒充已解决事项：

- 实际起跑的gate基线分支、快照与配置。main普通MLP、feature-selection入口和后续有效gate研究版本不同，不把任意一个旧版本自动提升为当前实验默认
- score 的绝对值位置、局部到全局聚合、非负scaling映射与刷新频率，保证开发和确认使用同一读出
- PQI实际目标、零点定义、维度归一化、c进入目标的位置及对应求解器，不能把L1软阈值当成完整比值近端
- 各方法A/B数值搜索范围、路径长度、原版停止设置与公平预算，B的100次搜索和10次初始化须转为实际配置并核对缺失格子
- Localized Lasso的任务适配和全局读出，STG/E2E-FS入口、DeepPINK输入依赖，候选三全局状态及批量缩放
- LassoFlexNet当前实现缺失、印刷算法2与hierarchy的待核一致性、TabZilla表格任务清单差异。保留方法必比身份，但代码未取得前记为待复现，不伪造结果

这些待核项决定能否开跑对应实验，不影响将两块研究问题、数据与选模协议作为计划发布。公开方法的完整版本、论文阅读范围、算法2的字面反例、旧代码清单及数据下载入口保留在 [Deep Regularization 四个候选相近文献](<调研 Search/文献调研 Literature Search/文献调研 Literature Search Deep Regularization.md>)。

主要执行依据：[本项目固定benchmark入口](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/main-benchmark.py)、[既有研究证据](https://github.com/DreamSoul-AI/Deep-Regularization/tree/e0626201327410f20037940658fec0ad60549273/docs/research/20260920-xor-ring-thread)、[十二真实数据集原论文](https://papers.neurips.cc/paper_files/paper/2023/file/82f39c7409155b74d15d73b048f06771-Paper-Datasets_and_Benchmarks.pdf)、[其数据与完整管线代码](https://github.com/vcherepanova/tabular-feature-selection)、[LassoFlexNet官方论文](https://proceedings.mlr.press/v306/lui26a.html)、[LassoFlexNet官方仓库](https://github.com/BorealisAI/lassoflexnet)。


## 沟通与同步入口

11月28日的 [Ranking Baseline 全量对照](<研究记录 Research Record/2026年11月28日 Ranking Baseline 全量对照.md>) 记录本轮实际设置、完整结果入口、Localized Lasso 的适配与数值核验，以及 LassoFlexNet 论文追加的比较清单。它是A部分128维、四任务六折的固定设置运行，不替代B部分12数据集的调参与下游协议。Localized Lasso 的任务适配及 STG/E2E-FS 等接口已在该范围内落实；LassoFlexNet 官方实现和 mRMR 编译依赖的限制仍须保留。

[AI 沟通流程](https://dreamsoul.feishu.cn/docx/BXz0dmV0QoNkjyxcO6Ec6ijencm) 保留讨论和决定的演进。完整沟通原件、用户配置、工具回执及论文 PDF 不随此文档提交。
