# 文献调研 Literature Search Deep Regularization

本报告保留定向文献与源码核对依据。最新研究执行口径以 [研究计划](<../../研究计划 Research Plan Deep Regularization.md>) 为准，A 的特征排名与 B 的下游预测均为正式实验。

## 一、结论与四个候选的对应关系

检索日期为2026年11月27日。本轮是围绕已讨论的四个候选做定向近邻筛选，目的是确定直接阅读对象、可复用部件与需比较的差异，不以“没搜到同名方法”认定创新。

| **候选** | **最接近的资料** | **已有重合** | **本方案仍需单独核验的部分** |
|---|---|---|---|
| 一：共享 gate，score 加入一阶与经验 Fisher 二阶信息 | Taylor pruning [1]、Group Fisher [2]、Reweighted ADMM [14] | 用删除损失估计重要性，逐样本梯度平方，重要性加权 ADMM 已有相关工作 | 保留一阶项、曲率代理及其符号口径，score 仅作固定 scaling，具体加权 PQI 子问题 |
| 二：普通前向，稀疏坐标使用 gate×输入权重列范数 | SPINN [3]、Network Slimming [4] | 用输入列或通道尺度表达结构稀疏 | 对乘积幅度建立 ADMM 一致性，真实剪枝时同步清零 gate 和输入列，避免前向归一化 |
| 三：保存独立 per-sample gate，局部 score 再聚合排名 | Localized Lasso [5]、LSPIN [6]、L2X [7]、INVASE [8] | 逐样本变量选择已有成熟先例，独立局部参数也已有先例 | 独立 gate 参数表＋共享非线性模型＋整表 PQI-ADMM，按删除 score 读出局部及整体排名 |
| 四：LassoNet 先转 ADMM-Lasso，再换 PQI | LassoNet [9]、ADMM pruning [10]、L1/L2-ADMM [11]、LassoFlexNet [12] | 层级约束、神经网络 ADMM 与比值正则 ADMM 的部件分别已有 | 把稀疏副本与 hierarchy 副本拆开后的一致性、实际模型可行性及多分类比值求解 |

本轮重点对标调整为 LassoFlexNet（ICML 2026）与 Localized Lasso（AISTATS 2017）。前者作为候选四的近期结构及优化对照，后者作为候选三独立局部参数的直接先例。Taylor/Fisher 与列稀疏文献保留作部件依据。具体机制与待核问题见第九章。

## 二、候选一：Taylor、Fisher 与加权 ADMM

Molchanov 等的 Taylor pruning 明确引入固定为 1、**不参与训练**的 gate 来计算重要性。论文 Eq.7–9 区分“组内贡献求和后平方”和“每个元素平方后相加”，二者并不等价。因此，用一个 gate 表示 feature，已经把它后面的多个权重贡献合进该 gate 导数，不能再任意把平方顺序交换。[1](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)

Group Fisher 在 §3.1 Eq.3 使用逐样本 mask 梯度的平方平均，§3.2 进一步处理耦合通道。这直接支持“先逐样本平方，再跨样本平均”的实现口径。论文主要从接近收敛的位置推导剪枝评分，并采用迭代剪枝与微调，未采用我们的可学习输入 gate 或 PQI-ADMM。[2](https://proceedings.mlr.press/v139/liu21ab.html)

本候选保留的探索是：令 $t_{nj}=g_j\partial\ell_n/\partial g_j$，使用下式的原始有符号量，再另行固定绝对值和正 scaling 映射：

$$s_j=\frac1N\sum_{n=1}^N\left(-t_{nj}+\frac12t_{nj}^2\right).$$

这里的第二项是删除 gate 时的经验 Fisher 曲率代理，不能称为精确 Hessian。最小化总损失时，平均梯度可以相消，而梯度平方平均仍非零。若每条样本的相关梯度都接近零，该代理也可能消失。一次删除到零是有限扰动，局部近似是否有效应与真实删除损失比较。

Reweighted ADMM（2024）的公开摘要说明它构造动态重要性矩阵并纳入 ADMM，已经与“重要性驱动缩放”接近。本轮未取得正文，暂不能断言它是否采用同一阶加二阶评分、同样的 gate 坐标或同一个 PQI 子问题。[14](https://doi.org/10.1016/j.neunet.2024.106534)

因此，论文定位不能只写“引入二阶信息”或“重要性加权 ADMM”。应准确说明评分对应的删除对象、保留一阶项的条件、scaling 接入位置，以及这些设计对 feature 排名与剪枝损失的影响。score 在本方案里仍是固定缩放，不新增一个可训练的 score 变量。

## 三、候选二：有效幅度与结构稀疏

SPINN 在 §2 Eq.4 对首层的每个输入列施加 sparse-group penalty，混合列内 L1 与列 L2，以产生整个 feature 的稀疏。它提供“输入列是选择单位”的直接参照，而不是对每个样本的数据列做优化。[3](https://arxiv.org/abs/1711.07592v2)

Network Slimming 使用 BatchNorm 的通道尺度系数及稀疏约束，并按通道剪枝。它说明尺度参数可以承担结构选择，但依赖 BN 的参数化，不能据此把候选二改成 forward weight normalization。[4](https://openaccess.thecvf.com/content_ICCV_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)

我们的有效首层为 $A=W\operatorname{diag}(g)$，所以 $q_j=|g_j|\lVert W_{:,j}\rVert_2=\lVert A_{:,j}\rVert_2$。若使用带符号的 $g_j\lVert W_{:,j}\rVert_2$，其绝对值仍等于该幅度。惩罚 $\sum_j q_j$ 与对有效首层按输入列做 Group Lasso 的几何意义接近。这是代数关系，不代表两种参数化的优化轨迹相同。

它的价值是避免仅缩小 $g_j$、同时放大 $W_{:,j}$ 就降低幅度惩罚。$g_j\to c_jg_j$、$W_{:,j}\to W_{:,j}/c_j$ 不改变 $q_j$ 的绝对值。这个结论限于该 gate 与紧随其后的线性列，不能据此外推整个多层网络对所有重参数化都不敏感。

需把**幅度坐标**和**删除 score**分开。前者决定 ADMM 对谁建立一致性，后者决定缩放强度。简单再给原来的 Taylor score 乘一个权重范数，不自动获得不变性。ADMM 使用 $Z=q(g,W)$ 时，模型块必须通过这个乘积同时向 gate 和权重回传。最终永久剪枝同步清零 gate 和整列权重，辅助变量为零本身不能证明模型路径已经删除。

## 四、候选三：独立样本参数的先例与局部重要性

**Localized Lasso 是必须补进来的直接参照。** 它在 §2.1–2.3 为每条样本学习独立线性系数向量，并用样本图上的网络正则和逐样本 exclusive 稀疏项连接这些局部模型。它的局部参数不是由共享 selector 网络生成，所以“每条样本存一组参数”本身已有先例。[5](https://proceedings.mlr.press/v54/yamada17a.html)

LSPIN 则在 §3 用共享 gating network 预测每条样本的随机 gate，并联合训练预测模型。它还加入相近样本 gate 的稳定性正则。与候选三相比，区别是 gate 的参数来源、随机 L0 松弛及样本间约束，而不是是否能给出 per-sample importance。[6](https://proceedings.mlr.press/v162/yang22i.html)

L2X 通过互信息准则训练逐样本特征选择函数，使用连续松弛学习子集，是局部解释的参照。[7](https://proceedings.mlr.press/v80/chen18j.html) INVASE 同样研究 instance-wise variable selection，本轮仅核到正式会议入口和公开摘要，其 selector、predictor 与 baseline 的具体更新尚未逐项核对，暂不作为已复核的实现依据。[8](https://openreview.net/forum?id=BJg_roAcK7)

我们的候选仍按原约定：保存 $G\in\mathbb R^{N\times d}$，由稳定 sample ID 索引，与共享模型一起优化。ADMM 的 $Z$ 与 $G$ 同形，score 仅用于 scaling。若目标是整表 PQI，就对整个矩阵的明确向量化定义求解，不能把每个 mini-batch 的比值之和当成同一个目标。

最终先保存每个 sample-feature 的删除 score，再沿样本维求和形成整体排名。局部 $G_{nj}=0$ 不能直接删除所有样本共用的 $W_{:,j}$。评估重点是局部重要性、聚合排名及其对真实删除影响的解释能力。用户当前不以新样本泛化为主要目标，不能用缺少共享 selector 的泛化能力直接否定此候选。

同时需要固定“解释哪一个模型”：联合训练 $G,W$ 得到的是适配后的模型重要性。如果要解释最初的原模型，可以增加固定模型、只拟合 gate 的对照。两种目标各自有效，不能把适配后得到的 score 默认为原模型的归因。这是后续评价口径建议，未改动当前候选定义。

## 五、候选四：LassoNet、ADMM 与 L1/L2 的组合

LassoNet 的 JMLR 完整版给出线性 skip、首层 hierarchy 与联合近端算法。原版的稀疏处理和层级限制是联动的，不能只换线性系数的阈值、再忽略非线性路径。[9](https://jmlr.org/papers/v22/20-848.html)

ADMM pruning 在 §4.2 将神经网络任务损失与稀疏约束拆分，通过辅助变量和对偶量迭代。原文采用基数约束的投影，保留指定数量的大幅度权重，属于硬选择。它支持 ADMM 拆分框架，但其投影不是 L1 的 soft-thresholding，更不是 PQI 的精确近端。[10](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.php)

Rahimi 等在 §4.1 将 L1/L2 的分子、分母放到两个一致性副本，并交替更新。这里确实会出现带分母缩放的软阈值子步骤，同时还必须更新分母副本。这说明“某个子步骤有 soft-thresholding”可以成立，但“整个比值的近端就是一次普通软阈值”不成立。其应用是带线性等式约束的稀疏恢复，不包含神经网络任务损失与 LassoNet hierarchy，求解和理论结论不能整套直接移植。[11](https://doi.org/10.1137/18M123147X)

LassoFlexNet（ICML 2026）在 §4.1–4.2 把输入特征先变为各自的嵌入，再用 tied skip 系数控制整个特征嵌入的非线性通路，并调整层级近端优化。这是近期需要加入的 LassoNet 系列参照。但本轮所读机制不能证明它采用我们的 PQI 或 ADMM 拆分。[12](https://proceedings.mlr.press/v306/lui26a.html)

因此仍按已确定路线：先保留 Lasso 和 hierarchy，仅改成 ADMM，核对预测模型的一致性残差与层级违反量。再替换 PQI 稀疏子问题。多分类时先按输入列取得类别权重的幅度，再对幅度向量定义比值。比值会耦合 feature，不能默认逐列独立阈值仍然精确。

## 六、阅读顺序与证据边界

| **顺序** | **资料** | **先解决的问题** |
|---|---|---|
| 1 | Group Fisher [2] 与 Taylor pruning [1] | 逐样本梯度如何累计，先求和还是先平方，gate 导数代表什么 |
| 2 | Localized Lasso [5] 与 LSPIN [6] | 独立局部参数与共享生成网络有什么实质差别 |
| 3 | LassoNet [9] 与 L1/L2-ADMM [11] | 层级一致性与比值求解分别怎样处理 |
| 4 | SPINN [3] 与 Network Slimming [4] | 输入列幅度及尺度稀疏有哪些已有机制 |
| 5 | LassoFlexNet [12] 与 Reweighted ADMM [14] | 补齐近期方法与加权 ADMM 的细节比较 |

本轮对新取得的 11 篇 PDF 核对首页、摘要与指定方法段。Taylor 重点读 Eq.7–9，Group Fisher 重点读 §3，SPINN 重点读 §2，Localized Lasso 重点读 §2，LSPIN 重点读 §3，L2X 重点读 §2–3，LassoNet 重点读 §3–4，ADMM pruning 重点读 §4.2，L1/L2 重点读 §4.1，STG 重点读 §2–3，LassoFlexNet 重点读 §4.1–4.2。未逐篇完整精读、复现或核验所有证明与实验。

OBD、Network Slimming 和已有 L1/L2 分析稿复用既有馆藏及既有机制依据，本轮未重新完整精读。INVASE 的 OpenReview PDF 下载返回 403，未取得可入库文件。Reweighted ADMM 的出版页正文访问受限，当前只有摘要和书目。二者保留入口，不计入新增 PDF。

检索覆盖 Taylor/Fisher pruning、输入尺度与列稀疏、instance-wise selection、独立局部模型、LassoNet 变种以及 L1/L2-ADMM。检索了 LassoNet 与 ADMM、PQI、L1/L2 等组合，并区分同名统计网络估计工具。本轮是定向筛选，未做穷尽式系统综述，也不能排除使用不同术语的相同组合。未来是否构成方法贡献，应在补齐受限正文后按目标、参数化、scoring、子问题和评价逐项比较。

价值等级按本课题的直接可用性给出，A+ 表示优先核对的直接方法或核心部件，A 表示重要对照，A- 表示有用的部分机制。它们不表示渠道等级或论文整体质量。引用量为 Semantic Scholar 在2026年11月27日的返回值，可能合并或分开预印本与正式版，不能跨版本直接当成统一排名。STG 返回了不同题名的旧预印本条目，暂不填作正式论文引用量。

| **资料** | **引用量** | **来源与口径** |
|---|---:|---|
| Importance Estimation for Neural Network Pruning | 1271 | [Semantic Scholar](https://www.semanticscholar.org/paper/a6f4917d043494d2ebaebe6b65cb35e6a07fda41)，数据库年份 2019，不替代正式发表年份 |
| Group Fisher Pruning for Practical Network Compression | 283 | [Semantic Scholar](https://www.semanticscholar.org/paper/2dcc6d63e840217306ecdef527ca66706eddc2cb)，数据库年份 2021，不替代正式发表年份 |
| Feature Selection using Stochastic Gates | 待核 | 返回不同题名旧预印本，未混用 |
| Locally Sparse Neural Networks for Tabular Biomedical Data | 51 | [Semantic Scholar](https://www.semanticscholar.org/paper/d88dfd5c534cb3b8913fa918d7a0d74c02bc8e52)，数据库年份 2021，不替代正式发表年份 |
| Learning to Explain: An Information-Theoretic Perspective on Model Interpretation | 700 | [Semantic Scholar](https://www.semanticscholar.org/paper/7ccc073c58f9be247eaa56bae37bea652c17ce29)，数据库年份 2018，不替代正式发表年份 |
| INVASE: Instance-wise Variable Selection using Neural Networks | 待核 | 本次未返回可核实匹配 |
| LassoNet: A Neural Network with Feature Sparsity | 229 | [Semantic Scholar](https://www.semanticscholar.org/paper/0ac17248c8e62ad539f53a84ed47a2409d06707b)，数据库年份 2019，不替代正式发表年份 |
| A Scale-Invariant Approach for Sparse Signal Recovery | 待核 | 本次未返回可核实匹配 |
| A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers | 481 | [Semantic Scholar](https://www.semanticscholar.org/paper/8f880afbfea328c496d1ab96a123bb57d4b506b5)，数据库年份 2018，不替代正式发表年份 |
| Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification | 112 | [Semantic Scholar](https://www.semanticscholar.org/paper/b70aaa8506989c7026a3ccd7bcee2bb4b4318d7f)，数据库年份 2017，不替代正式发表年份 |
| LassoFlexNet: a Flexible Neural Architecture for Tabular Data | 待核 | 本次未返回可核实匹配 |
| Localized Lasso for High-Dimensional Regression | 55 | [Semantic Scholar](https://www.semanticscholar.org/paper/6eaf4d5f36610f3a127f9a9d4a360d72804cc12b)，数据库年份 2016，不替代正式发表年份 |

## 七、论文入口与收录版本

1. Pavlo Molchanov、Arun Mallya、Stephen Tyree、Iuri Frosio、Jan Kautz。[Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)。CVPR 2019。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/G5oGb34V2oIM3Vxn9F8ce7JunAd)，价值 A+。

2. Liyang Liu、Shilong Zhang、Zhanghui Kuang、Aojun Zhou、Jing-Hao Xue、Xinjiang Wang、Yimin Chen、Wenming Yang、Qingmin Liao、Wayne Zhang。[Group Fisher Pruning for Practical Network Compression](https://proceedings.mlr.press/v139/liu21ab.html)。ICML 2021。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/Du7vbqEOUoYMH4xl3l4c977anfb)，价值 A+。

3. Jean Feng、Noah Simon。[Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification](https://arxiv.org/abs/1711.07592v2)。arXiv 2019。arXiv:1711.07592v2，2019修订，2017首发，未核实正式发表渠道。[飞书 PDF](https://dreamsoul.feishu.cn/file/I3Ymbp2VqottGDxSVI1cJR2WnBS)，价值 A-。

4. Zhuang Liu、Jianguo Li、Zhiqiang Shen、Gao Huang、Shoumeng Yan、Changshui Zhang。[Learning Efficient Convolutional Networks through Network Slimming](https://openaccess.thecvf.com/content_ICCV_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)。ICCV 2017。复用既有馆藏，本轮按既有机制依据作对照。

5. Makoto Yamada、Koh Takeuchi、Tomoharu Iwata、John Shawe-Taylor、Samuel Kaski。[Localized Lasso for High-Dimensional Regression](https://proceedings.mlr.press/v54/yamada17a.html)。AISTATS 2017。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/XOLNbo5gLo1LMrxGFsFcWpHNnGe)，价值 A+。

6. Junchen Yang、Ofir Lindenbaum、Yuval Kluger。[Locally Sparse Neural Networks for Tabular Biomedical Data](https://proceedings.mlr.press/v162/yang22i.html)。ICML 2022。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/GbYZbifpeo7zk3xUxsrcSM0znRh)，价值 A+。

7. Jianbo Chen、Le Song、Martin J. Wainwright、Michael I. Jordan。[Learning to Explain: An Information-Theoretic Perspective on Model Interpretation](https://proceedings.mlr.press/v80/chen18j.html)。ICML 2018。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/YW53bKYKToAAWsxlWy5cG3iqnKb)，价值 A。

8. Jinsung Yoon、James Jordon、Mihaela van der Schaar。[INVASE: Instance-wise Variable Selection using Neural Networks](https://openreview.net/forum?id=BJg_roAcK7)。ICLR 2019。ICLR 2019，PDF下载403，未入库。PDF 未取得，保留正式入口。

9. Ismael Lemhadri、Feng Ruan、Louis Abraham、Robert Tibshirani。[LassoNet: A Neural Network with Feature Sparsity](https://jmlr.org/papers/v22/20-848.html)。JMLR 2021。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/OP93blGVIoGzJAxMaipcckLun1c)，价值 A+。

10. Tianyun Zhang、Shaokai Ye、Kaiqi Zhang、Jian Tang、Wujie Wen、Makan Fardad、Yanzhi Wang。[A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.php)。ECCV 2018。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/CGOtbJN52oG5KAxExWwcQiJ8nyg)，价值 A。

11. Yaghoub Rahimi、Chao Wang、Hongbo Dong、Yifei Lou。[A Scale-Invariant Approach for Sparse Signal Recovery](https://doi.org/10.1137/18M123147X)。SIAM Journal on Scientific Computing 2019。UCLA CAM 19-02 公开作者稿，非出版社排版版。[飞书 PDF](https://dreamsoul.feishu.cn/file/BugubtgmGo1igMx5ltccHojWnih)，价值 A+。

12. Yik Chau Lui、Cheng Chi、Kishore Basu、Yanshuai Cao。[LassoFlexNet: a Flexible Neural Architecture for Tabular Data](https://proceedings.mlr.press/v306/lui26a.html)。ICML 2026。ICML 2026 官方会议 PDF，作者首名PDF与PMLR写法不同，以PMLR书目信息索引。[飞书 PDF](https://dreamsoul.feishu.cn/file/Ua0DbjrPLo0JHkxtPoUcXAETnog)，价值 A。

13. Yutaro Yamada、Ofir Lindenbaum、Sahand Negahban、Yuval Kluger。[Feature Selection using Stochastic Gates](https://proceedings.mlr.press/v119/yamada20a.html)。ICML 2020。官方会议或期刊版本。[飞书 PDF](https://dreamsoul.feishu.cn/file/RAtxbqJ4Ooz8fhxMTcAc9rsSnwg)，价值 A。

14. Ming Yuan、Lin Du、Feng Jiang、Jianchao Bai、Guanrong Chen。[Reweighted Alternating Direction Method of Multipliers for DNN weight pruning](https://doi.org/10.1016/j.neunet.2024.106534)。Neural Networks 179:106534，2024。摘要层候选，正文未取得，未入库。

补充基础资料：[Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)，Yann LeCun、John S. Denker、Sara A. Solla，NeurIPS 1989，复用既有 PDF。[Analysis of The Ratio of l1 and l2 Norms in Compressed Sensing](https://arxiv.org/abs/2004.05873v2)，复用既有 arXiv v2 条目，未按同一论文的未核实正式版改名。

## 八、本次资料同步范围

本地统一论文库新增 11 篇，已核对真实 PDF、题名、版本、最终哈希与索引路径。Feature Selection 课题新增 8 篇，Pruning 新增 3 篇。既有 OBD、Network Slimming 与 L1/L2 分析稿不重复下载或重建。

11 篇均已上传到飞书统一论文库对应目录，逐篇下载回读，与本地 SHA256 一致，并重新核对父目录。上面的“飞书 PDF”可以直接打开。项目资料通过这些链接复用统一馆藏，没有再复制一套项目 PDF。

本地课题索引、总账与总导航已增量更新。本轮云端交付为新增 PDF 与本报告，既有云端原生 Markdown 索引和总账未覆盖更新。百度同步盘本地文件已写入和回读，未核验百度客户端的服务端上传完成状态。

本报告同步到已核实的 Deep Regularization 项目目录，具体云端标识、正文回读、双侧基线和逐篇文件回执保留在本次同步记录。本报告的相近方法结论补入 [输入门控与 LassoNet](<../通用调研 General Search/通用调研 General Search Deep Regularization 输入门控与 LassoNet.md>) 的资料入口。调研阶段未修改模型代码或恢复训练，本次同步仅提交研究文档。

## 九、LassoFlexNet 与 Localized Lasso 重点对标

本节于2026年11月27日补充定向方法阅读。LassoFlexNet 阅读范围扩展至 §4–7、算法 1–2 与 Table 2、§7.1 特征恢复评价，未复核附录全部证明。Localized Lasso 重点核对 §2.1–2.3、Eq.1–5 与 Algorithm 1，未完整重读所有实验。下列代数对应、反例和对照建议为本报告分析，不冒充作者结论。

### （一）LassoFlexNet：优先比较结构与更新机制

它先对每个原始 feature 单独编码与学习嵌入，再让共享 Mixer 建模交互。skip 系数作用于该 feature 的平均池化嵌入，层级限制则绑定其完整嵌入块的出边。因此，“一个 feature 对应多个 embedding 坐标”没有变成“多个可独立选择的原始 feature”。这与我们的原始输入 gate 不同，比较时要对齐最终删除的原始列。[论文 §4](https://proceedings.mlr.press/v306/lui26a.html)

它还在嵌入末端使用关闭可学习 affine 参数的 BatchNorm。这个处理直接进入 forward，不等于仅在评分时抽取范数。因此，可把它列为归一化参照，但不能声称它与候选二保持同一前向和同一更新几何。

优化部分保留层级思想，优先更新 skip 选择信号，再处理首层，并加入 Adam 的逐坐标阈值与参数 EMA。它不是把普通 Lasso 直接换成另一种稀疏惩罚。Table 2 同时变化了多个组件，单数据集消融不能自动证明每个组件在全部任务都有优势。正文 §6 的理想化分析也明确不构成完整 Seq-Hier-Prox-Adam-EMA 的端到端收敛证明。

**算法 2 有一处应先向实际实现核对的约束疑点。** 按印刷公式逐项计算，取单元素出边、$M=1$、$\lambda=1$、$\beta_{\mathrm{EMA}}=0$、$W_{\mathrm{EMA}}=3$。第 3 行给出 $\widetilde\beta=0$，第 6–7 行选中 $m=1$ 时边界 $w_1=\tfrac12 S_1(3)=1$，第 8 行留下 $\widetilde W=1$。这不满足正文声明的 $|\widetilde W|\le M|\widetilde\beta|=0$。反例仅针对印刷算法的字面更新，不据此断言作者实际代码或全部实验错误。需核对是否省略了后续裁剪、skip 回写或其他约定，不能自行补步骤后称为忠实复现。

截至本轮核对，[官方仓库](https://github.com/BorealisAI/lassoflexnet) 仅见 README，写有“code coming soon”。因此当前可做论文公式对标，尚无可核验的公开训练实现。本轮没有运行或复现该方法。

评价应拆成两层：同一原始特征集合、划分、骨干和预算下比较稀疏机制，再单列 PLE、PFE、Mixer、BN、顺序更新及 EMA 的贡献。候选四仍先做原 LassoNet 的 ADMM-Lasso，再换 PQI，同时把 LassoFlexNet 列为近期主参照，不把其整套结构收益归因给 Lasso 或 PQI。

§7.1 的恢复实验以原始输入与添加的纯噪声区分 feature，使用 skip 幅度排名。该标签说明对注入噪声的区分能力，不等于已知每个原始变量的真实因果重要性。我们的梯度删除 score 还需单独对照真实删除损失，不能仅用预测分数替代 importance 评价。

### （二）Localized Lasso：与独立 per-sample gate 的对应

普通 Lasso 全部样本共享一个向量 $w$。Localized Lasso 为样本 $n$ 单独保存 $w_n\in\mathbb R^d$，预测为 $\widehat y_n=w_n^\top x_n$。这就是它与候选三最直接的联系：两者都在样本和 feature 两个维度上保存可调参数，允许不同样本选择不同特征。[论文 §2](https://proceedings.mlr.press/v54/yamada17a/yamada17a.pdf)

其平方损失形式为：

$$\min_{\{w_n\}}\ \sum_n(y_n-w_n^\top x_n)^2+\lambda_1\sum_{n,m}r_{nm}\lVert w_n-w_m\rVert_2+\lambda_2\sum_n\lVert w_n\rVert_1^2.$$

第一项拟合各样本。第二项使图上相近样本的局部系数靠近。第三项在**每条样本内部**做 exclusive 稀疏，其展开为：

$$\lVert w_n\rVert_1^2=\left(\sum_j|w_{nj}|\right)^2=\sum_jw_{nj}^2+2\sum_{j<k}|w_{nj}w_{nk}|.$$

这里是先对一行取 L1 再平方，不是先对一列跨样本取 L2，也不是 L1/L2。组内的交叉乘积使多个 feature 在同一样本内产生竞争。图项再把不同样本连接起来，不能把它概括成互不相关地拟合 N 个普通 Lasso。

用线性预测器看我们的方法，若共享权重为 $v$，则 $\widehat y_n=v^\top(g_n\odot x_n)=(v\odot g_n)^\top x_n$。因此可将有效局部系数记作 $w_n=v\odot g_n$。当共享坐标非零且 gate 为不受限实数时，这个表示可以覆盖任意局部系数；若 gate 限制为非负、区间或二值，表达范围相应受限。换成共享 MLP 后，gate 位于非线性模型前，已不能直接等同于原文的局部线性模型。

核心差异如下：

| **比较点** | **Localized Lasso** | **候选三** |
|---|---|---|
| 每样本保存量 | 直接预测用的局部系数 | 输入 gate，之后接共享模型 |
| 稀疏项 | 逐样本平方 L1 | 明确定义的整表 PQI，score 作固定 scaling |
| 样本关联 | 图上的系数距离正则 | 当前候选未自动加入图正则 |
| 读出 | 局部系数与稀疏模式 | 局部删除 score，再跨样本聚合排名 |
| 优化 | 论文采用迭代最小二乘 | 神经网络任务更新与 ADMM |
| 理论适用范围 | 固定设计矩阵下的凸目标 | 一般为非凸，不能继承其全局最优结论 |

作者在 §2.2 说明可用 ADMM，但选择自己的迭代最小二乘算法。“无需 tuning parameter”指该优化程序不引入相应的算法调参，并不表示没有稀疏强度：目标和 Algorithm 1 仍明确包含 $\lambda_1,\lambda_2$。

因此，第三候选应以它作为直接先例，但保留自己的研究问题：独立 gate 配合共享非线性模型时，PQI 与一阶加曲率的固定 scaling 能否得到更有用的局部重要性和聚合排名。评价可包含局部特征恢复与删除损失，不能仅因模型拟合训练数据就推断 importance 正确。本节未增加样本图正则，也未变更既定 ADMM 目标。

## 十、正式基线与评价口径

LassoFlexNet 与 Localized Lasso 均列入本课题正式 baseline，不再仅作为相关工作引用。这里登记的是必须接入的比较对象，尚未实现或运行不等于已经取得实验结果。

| **方法** | **基线用途** | **特征选择单位与读出** | **当前实现状态** |
|---|---|---|---|
| LassoFlexNet，ICML 2026 | 近期神经网络特征选择主对照，重点比较候选四 | 共享 skip 系数控制一个原始 feature 的完整嵌入通路，按原生系数幅度和支持集读出 | 已入库，论文方法已定向阅读。官方仓库 code coming soon，待取得实现并核对算法 2 与 hierarchy 的一致性 |
| Localized Lasso，AISTATS 2017 | 局部特征选择主对照，重点比较候选三 | 每个 sample 的局部线性系数及其稀疏支持，允许各样本使用不同 feature | 已入库，目标与算法已定向阅读。原实现的可用性、分类及多分类支持仍待核验 |

Localized Lasso 可以做 feature selection，原生任务是逐样本选择，并不要求所有样本共享同一组 feature。比较整体排名时，需要显式增加聚合读出。例如在一致的输入标准化下，可预先固定 $I_j=N^{-1}\sum_n|w_{nj}|$ 作为一个候选聚合规则，并同时报告每个 feature 的局部选中比例。这是本项目评价适配，不冒充原论文已有的统一全局 ranking。最终读出口径实施前固定，不能看结果后挑选。

共同评价至少区分两种问题：方法本身的局部或整体特征选择能力，以及选出的特征对指定旧网络的影响。后者可将各方法得到的支持或排名作用到同一个已固定的目标网络，比较相同删除数下的损失变化。若另外训练新模型评价保留特征的预测价值，应单独标记为重训结果。LassoFlexNet 和 Localized Lasso 各自重新拟合得到的重要性，不自动等于旧网络自身的归因。

数据划分、标准化、保留特征数或局部稀疏预算、调参预算和重复运行口径保持可比。局部选择报告 per-sample 支持与局部恢复指标，整体选择报告聚合排名、恢复指标及删除曲线。预测表现与 importance 质量分别报告。完整架构比较与同骨干下的正则消融分别记录，不能把 PFE、Mixer 或局部线性模型的差异都归因于稀疏惩罚。

原版 LassoNet 以及已定义的 ADMM-Lasso 仍保留为候选四的机制参照。新增两篇基线不会替换现有四候选，也不代表已开始训练、已实现算法适配或已复现论文性能。

## 十一、仓库既有基线的代码核对

2026年11月27日补充。新增 LassoFlexNet 与 Localized Lasso 应接入已有 baseline 体系，不能把它们列成全部对照。当前 checkout 为 main `e0626201327410f20037940658fec0ad60549273`，其 src 不是完整特征选择 benchmark。以下实验代码固定到 feature-selection `555ba11e8872bd6618cd721bd3b08f7128db9e6b`，另用 main 保留的研究快照与历史表核对执行证据。只读源码和既有结果，没有运行训练或新增实验。

### （一）已存在的方法

| **类别** | **已有方法或入口 ID** | **代码及结果状态** |
|---|---|---|
| 传统特征选择 | Random Forest（rf）、Mutual Information（mi）、Relief（relief）、mRMR（mrmr） | benchmark 的 core 分派和命令入口均存在，保留的128维历史表有相应结果 |
| 神经网络特征选择 | LassoNet（lassonet）、FSNet（fsnet）、DeepPINK（deeppink）、Concrete Autoencoder（cae） | benchmark 有调用分支，历史表有结果。DeepPINK 还依赖对应 knockoff 输入 |
| 输入门控 | CancelOut sigmoid（canceloutsigmoid）、CancelOut softmax（canceloutsoftmax） | 两个变体均有实现和历史结果，softmax 保留历史 secondary 身份，不提升为默认主方法 |
| 全局随机门控 | Stochastic Gates（stg） | 仓库根 src/core.py 已接 run_stg_fs，包装器调用 stg 包并读取 gate probability。嵌套 benchmark 的命令注册未包含 stg，不能说它已进入该入口的全部实验。128维历史表没有STG行 |
| 端到端选择 | E2E-FS（e2efs） | 嵌套 benchmark 有命令、core 分支和包装器，包装器依赖 vendor E2EFS。根 src 也保留包装文件，但所读根 core 没有 e2efs 分派。128维历史表无此结果，不用“有文件”代替“已有完整实验” |
| 模型归因 | Saliency、Input×Gradient、IG_noMul、SmoothGrad、GuidedBackprop、DeepLift、Deconvolution、FeatureAblation、FeaturePermutation、ShapleyValueSampling | run_fa_method 中已有10个 Captum 调用分支，历史表均有结果 |
| 树模型归因 | TreeSHAP（treeshap） | 放在 run_fs_method 分派中，但机制属于归因。代码重新拟合随机森林，再读取 SHAP 并聚合，历史表有结果 |
| 控制与参考 | 普通 NN（nn）、随机分数（random） | nn 是已注册的预测控制；random 是 core 内的辅助分支，未列入所读命令 choices，不能认定为同入口的已运行 baseline |

历史128维表共有25行方法，其中包括21个外部对照和4条本研究或legacy行。该统计是已有历史对照表的覆盖范围，不是本次重新跑了25种方法，也不包含新增两篇基线。E2E-FS、STG 和新增两篇的缺失结果不得用其他版本结果补填。

### （二）入口与真实评分路径

[benchmark 主入口](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/main-benchmark.py) 的 FS_METHODS、SA_METHODS 与命令 choices 决定可选方法。process_dataset 分别走普通特征选择或 attr、attr-t、attr-b 路径，再进入 [benchmark core](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/src/core.py)。这些名称是注册和分派证据，仍需结合依赖与结果核对运行状态。nnfs 虽出现在注册表中，但不在所读命令 choices，也没有对应 core 分支，不能计作另一套已闭合实现。

归因分支先取得 wrapper 模型的逐样本 attribution，再用 mean(abs(attr), axis=0) 聚合成每个 feature 的整体分数。所读调用显式使用 target=0，解释的是指定模型输出，不应直接写成我们的监督损失一阶加曲率 score。attr、attr-t、attr-b 分别涉及测试样本、训练样本与 bootstrap 模型等口径，这些结果不宜不加说明地拼成同一个比较。

共同读出由 run_method 对绝对分数排序，并计算 best-k、best-2k 的真实特征恢复。best-k 是命中数除以真实相关特征数，不能因为名称叫 best-k 就当作预测 accuracy。LassoNet 的 feature_importances_ 来自稀疏路径，预测模型则按路径 validation loss 选择状态，重要性与预测状态不是同一个简单终点权重。

尤其要区分同一分支的两个 src：[根 core](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/core.py) 接入自有 ADMM 与 STG，[STG 包装器](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/stg_wrapper.py) 读取全局 gate；嵌套 benchmark core 保留 E2E-FS。不能切换工作目录或导入路径后默认方法集合仍然一致。实际调用入口还须固定 Python 模块解析位置。

### （三）历史结果与后续基线接入

[已有128维 baseline 表](https://github.com/DreamSoul-AI/Deep-Regularization/blob/e0626201327410f20037940658fec0ad60549273/docs/research/20260920-xor-ring-thread/baseline_128_table.md) 与对应研究说明将外部 baseline 标记为历史、非配对参照。它们的数据实例、预处理、划分与训练预算未统一到后续 Base/Five 研究线。本次不把这些旧数值提升为四候选的配对结果。

后续基线矩阵保留上表全部既有方法，新增 LassoFlexNet 与 Localized Lasso，分别记录代码接口、依赖、选择或归因单位、已有结果和待运行格子。STG、E2E-FS 的入口衔接与依赖核验列为接入事项。Sequential Attention 在已有研究记录中出现过，但本轮所读核心分派未定位到闭合调用，不标为已可直接运行。

对“接入旧网络”的研究问题，优先复用既有归因接口，并明确所解释的目标模型、输出或损失、样本范围及聚合方法。完整特征选择模型仍保留为支持恢复和下游预测对照。新增两篇是在这套基础上的补充，不能替代已实现的方法或重写历史实验身份。

## 十二、可直接承接的数据集与 Benchmark

本节保留与 LassoFlexNet 对标的数据与横向 benchmark 清单。两块正式实验分别评价已知支持下的 feature ranking 与十二真实数据集的下游预测效果。TabZilla 与 TabReD 用于后续预测扩展，不能仅因数据真实就认定其相关特征真值已知。机制诊断与消融保留为第十三章补充，任务口径见第十四章。

| **Benchmark** | **LassoFlexNet 的使用范围** | **我们可以承接的内容** | **建议优先级** |
|---|---|---|---|
| A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning，NeurIPS 2023 | 12个真实表格数据集，在额外特征干扰下评价选择与下游预测 | 数据、额外特征生成、原始划分和 MLP / FT-Transformer 下游评价管线 | 正式实验 B，不替代实验 A 的真值 ranking |
| TabZilla，NeurIPS 2023 | 论文自述使用36个 hard tasks 中的35个，Table 10 名单与该数量存在待核差异 | 官方36任务母集及 OpenML task ID，跨数据集的表格模型比较 | 第二优先，扩展真实数据覆盖 |
| TabReD，ICLR 2025 | 8个带丰富工程特征的数据集，随机与时间划分 | 官方预处理数组与划分，较大规模、混合类型与时间分布变化 | 第三优先，验证规模和实际应用条件 |

以下来源于官方论文、数据配置和项目下载说明，读取日期为2026年11月27日。没有下载完整数据、安装环境或执行实验。当前仓库的 Feature-Selection-Benchmark 与这里的 NeurIPS 2023 十二数据集 benchmark 是不同项目，不能因名称相近而认定已接入。

### （一）第一优先：12个真实数据集的特征选择 Benchmark

[官方代码与使用入口](https://github.com/vcherepanova/tabular-feature-selection)、[数据配置清单](https://github.com/vcherepanova/tabular-feature-selection/tree/main/config/dataset)、[原论文 Appendix B / Table 4](https://papers.neurips.cc/paper_files/paper/2023/file/82f39c7409155b74d15d73b048f06771-Paper-Datasets_and_Benchmarks.pdf) 给出如下名单。规模为原 benchmark 表中样本数与加入额外特征前的数值维度，实际装载时以所固定版本为准。

| **数据集** | **仓库配置名** | **样本数** | **原特征数** | **任务** |
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
| Microsoft | microsoft | 1,200,192 | 136 | 按该 benchmark 作为回归 |
| Otto Group Product Classification | otto | 61,878 | 93 | 9类分类 |
| Year | year | 515,345 | 90 | 回归 |

共8个分类、4个回归任务。Microsoft 原表规模为120万量级，但原论文 B.3 将实验样本限制到500,000，不能把表中原始规模当成全部实际训练量。House 在不同段落出现16K / 16H写法，本表按 Appendix B 与配置名 house 识别，执行时绑定实际来源和文件。配置名相同也不意味着不同仓库的样本子集相同。

**这套 benchmark 的关键是给同一份真实数据构造额外特征版本。** 上游 [data_tools.py](https://github.com/vcherepanova/tabular-feature-selection/blob/main/deep_tabular/utils/data_tools.py) 已有三种生成方式：random_feats 加独立高斯列，corrupted_feats 加原始列的加噪副本，secondorder_feats 加抽取原始列后的乘积。LassoFlexNet 的主比较表重点覆盖随机与二阶额外特征，原 benchmark 还提供损坏副本设置。

noise_percent 表示新增列占扩展后数值列的比例，不是相对原维度直接增加同样百分比。上游实现使用 $d_{extra}=\lfloor d\,p/(1-p)\rfloor$。因此50%对应约扩为2倍维度，75%对应约4倍。以 California Housing 为例，8维扩成16维或32维。按12个数据集、两种额外特征、两个比例计算是48个数据设置，加 corrupted 后为72个设置，不能写成48或72个独立数据集。

下载入口沿 [Revisiting Deep Learning Models for Tabular Data](https://github.com/Yura52/tabular-dl-revisiting-models#33-data) 和 [On Embeddings for Numerical Features](https://github.com/Yura52/tabular-dl-num-embeddings#data) 的数据包说明，放置位置按上游 loader 配置适配。原 loader 存在 /data 的固定路径，不宜直接覆盖我们现有目录。优先复用其预处理数据与索引，不凭同名 sklearn / OpenML 数据集替代原划分。上游 README 明确当前实现面向数值特征，此12数据集集合适合先避开类别列编码适配。

建议先接6个代表性数据集：California Housing、House 16H、Eye Movements、Gesture、Higgs 98k、Jannis。它们覆盖回归、二分类与多分类，样本和类别规模相对便于先跑通共享 gate 与 per-sample gate。随后补齐其余6个，形成完整十二数据集主表。这个6数据集子集是本项目的先行接入建议，不是原 benchmark 的官方简化版，也不预设哪些数据上本方法更有利。

该 benchmark 原生比较对象还包含 Deep Lasso、First-Layer Lasso、Adaptive Group Lasso、树模型与 Transformer attention 等。尤其 [Deep Lasso 代码](https://github.com/vcherepanova/tabular-feature-selection/blob/main/deep_lasso.py) 已对输入梯度做跨样本的 L2 聚合及特征维求和正则，与我们讨论的数据梯度有直接关系。接这套数据时，应承接其原生方法清单与我们已有 baseline，而不是只增加 LassoFlexNet 一条比较线。

### （二）第二优先：TabZilla 官方 Hard Benchmark

[官方仓库](https://github.com/naszilla/tabzilla)、[36任务的完整脚本和 task ID](https://github.com/naszilla/tabzilla/blob/main/scripts/HARD_DATASETS_BENCHMARK.sh)、[OpenML 官方任务集合](https://www.openml.org/search?type=study&study_type=task&id=379&sort=tasks_included) 可以直接定位数据。36个任务的显示名如下，最终执行身份以脚本中完整 OpenML task ID 为准：

Australian、Bioresponse、GesturePhaseSegmentationProcessed、MiniBooNE、SpeedDating、ada_agnostic、airlines、albert、artificial-characters、audiology、balance-scale、cnae-9、colic、credit-approval、credit-g、electricity、elevators、guillermo、heart-h、higgs、jasmine、jungle_chess_2pcs_raw_endgame_complete、kc1、lymph、mfeat-fourier、mfeat-zernike、monks-problems-2、nomao、one-hundred-plants-texture、phoneme、poker-hand、profb、qsar-biodeg、socmob、splice、vehicle。

这套主要提供广泛且有难度的真实分类任务，适合形成跨数据集横向结果。它不是每列自带相关性真值的特征选择库，不能直接把其中所有原始列都当作“应恢复的真特征”。若另行注入额外特征，应清楚标为我们的 TabZilla 扩展设置。

LassoFlexNet 自述去掉一个纯类别数据集后使用35个任务，但本轮对其 PDF Table 10 逐行计数并查看渲染页，实际列出33个命名数据行，而且出现 california。与官方36任务脚本不是一份可直接逐项匹配的清单。这里保留论文自述与表格现状，不自行补出缺失任务，也不把其汇总结果当成官方36任务的完整结果。正式复现其子集前需核对作者的 task ID、排除条件与缺失项。

数据通过官方 OpenML 预处理入口取得，例如 TabZilla/tabzilla_data_preprocessing.py 支持指定完整 dataset_name。当前主任务先固定数值或明确支持的混合类型子集，记录全部排除项，再扩展类别列处理。不能依据同名缩写，混用此处的 higgs 与上一套 Higgs 98k 的具体版本或样本。

### （三）第三优先：TabReD 的8个实际应用数据集

[官方仓库与当前下载说明](https://github.com/yandex-research/tabred/blob/main/readme.md) 已提供预处理数据和不同划分。下列数字按当前官方 README 的 Instances Used 与 Features 字段，不以原始可用全量数据代替该子集规模。

| **数据集** | **数据 ID** | **使用样本数** | **特征数** | **任务** |
|---|---|---:|---:|---|
| Homesite Insurance | homesite-insurance | 260,753 | 299 | 分类 |
| Ecom Offers | ecom-offers | 160,057 | 119 | 分类 |
| Homecredit Default | homecredit-default | 381,664 | 696 | 分类 |
| Sberbank Housing | sberbank-housing | 28,321 | 392 | 回归 |
| Cooking Time | cooking-time | 319,986 | 192 | 回归 |
| Delivery ETA | delivery-eta | 416,451 | 223 | 回归 |
| Maps Routing | maps-routing | 340,981 | 986 | 回归 |
| Weather | weather | 423,795 | 103 | 回归 |

它的主要价值是工程化特征较丰富，并保留时间分布变化。官方当前数据结构提供 default、random-{0,1,2}、sliding-window-{0,1,2} 的 train / val / test 索引，数组区分 x_num、x_cat、x_bin 与目标。可以直接比较随机划分与时间划分下的选择结果，不需要从原始业务记录重建整套数据工程。

当前官方推荐通过 tabred 下载器取得预处理版本，入口示例为 uvx tabred download，需按说明配置 Kaggle 身份与资源访问。这里仅提供操作入口，未安装下载器或触发数据下载。具体许可与获取条件以对应 Kaggle 来源及官方数据说明为准，不把可见仓库当成已经拿到了全部数据。

完整8数据集适合作为后续应用主表。先行验证时可从 Sberbank Housing 与 Ecom Offers 分别覆盖回归和分类，再扩展其余数据。独立 per-sample gate 在几十万样本、数百维设置中会显著放大参数和 ADMM 状态存储，需要单独记录数据规模，而非把不同方法在不同子样本上的结果并作同一个8数据集 benchmark。

### （四）对本项目的数据组合建议

主 ranking benchmark 保留已有 XOR / Ring 等已知支持任务。十二真实数据集另作下游价值补充，提供与 LassoFlexNet 对应的真实数据管线。TabZilla 扩展广度，TabReD 扩展规模与时间变化。三套 benchmark 不互相替代，也不将重复数据或多个额外特征比例算成独立新数据集。

接入前只需先固定数据版本、原始列与编码列的映射、任务类型和划分，随后对齐已有方法的输入输出。现有部分实验入口按二分类与特定预测形状编写，因此回归、多分类与混合类别列需要明确适配，不能认为下载数据后所有旧接口会自动支持。此处登记数据来源和建议范围，不新增下载、训练或复现实验授权。

## 十三、补充：机制诊断与消融评价建议

本节基于 ICML 2026 官方 PDF 的定向实验阅读：§6.1、Table 2、§7.1–7.4、附录 A、B.2–B.4、C.1–C.3、C.5，重点核对协议、选模与评价单位。不是复现实验，也未复核所有附录证明、逐数据集统计量或实现细节。[官方论文入口](https://proceedings.mlr.press/v306/lui26a.html)

### （一）总体判断

最值得借鉴的是分层证据：用已知结构的问题检查特征恢复，用真实数据检查保留信息和预测价值，用重复运行检查稳定性，再用训练轨迹解释优化机制。对本项目尤其重要的是再加一层：**对指定旧网络的忠实性**。真实相关变量恢复、解释旧模型依赖以及适配后新模型预测，是三个不同问题，分别立证。

旧模型可能已经依赖噪声，或没有学会真实交互。在这种情况下，忠实解释旧模型与恢复数据生成真值未必给出同一排名。不能因为一个方法选中了已知真实 feature，就直接认定它更忠实地解释了旧网络，也不能仅因删除损失增大就推断找到了因果变量。

### （二）原论文怎样做实验，能支持什么

| **证据类型与位置** | **论文实际做法** | **可借鉴之处与适用边界** |
|---|---|---|
| 组件消融，§6.1 / Table 2 | California Housing 上比较编码、嵌入、归一化、骨干与优化组件，报告10次初始化的均值、标准差和范围 | 同时看均值与波动值得采用。它是单数据集组件证据，且部分对照同时改变多个因素，不能等同于每个组件在全部任务上都有独立贡献 |
| 预训练后正则阶段，B.2 | 在多个超参数配置上对照预训练与 λ-training 的验证表现，允许选模回退到预训练状态 | 适合启发同起点比较，但额外训练预算和选模回退必须计入。“最佳验证结果不变差”受候选集包含起点影响，不能单独证明正则改善独立测试表现 |
| 调参敏感性，B.3 | 多个任务上采样超参数配置，画验证分数分布 | 比只报最优配置更有信息。窄分布也可能全部较差，需同时看性能水平、稀疏度与失败率，区分超参数变化和初始化变化 |
| 优化诊断，B.4 / C.1 | 已知线性信号的合成问题，以及同一预训练模型上的 skip 和首层权重变化 | 先构造能暴露机制的简单问题值得采用。但 C.1 的受控近端轨迹不等同于完整端到端训练收益，需与最终支持和预测行为连接 |
| 噪声下特征选择，§7.1 / C.2–C.3 | 在真实表格数据加入随机或相关二阶额外特征，比较不同选择器接 MLP / FT-Transformer 等管线 | “同一选择结果交给下游模型”适合我们的可移植目标。LassoFlexNet 自身同时更换表示和骨干，因此跨整套模型的差距不能只归因于选择算法 |
| 特征恢复与稳定性，§7.1 | 两个数据集加纯噪声，用 skip 幅度区分原始变量与额外噪声，报告恢复 AUROC、AP 和跨运行排名相关 | 恢复、稳定性分开报告值得保留。这里原始变量是正类标签，不代表每个原始变量都是真正相关或具有因果作用 |
| 广泛预测比较，§7.2–7.4 | TabZilla 的多数据集比较，TabRed 的随机与时间划分，以及树模型、深度模型和基础模型对照 | 可用于后期外部有效性与漂移测试，暂不需要把整个大规模预测榜单搬成首轮机制实验 |

关键协议也应保留原义。附录 C.2 的特征选择实验采用原 benchmark 的 0.65 / 0.15 / 0.20 训练、验证、测试划分，LassoFlexNet 做100次 Optuna 搜索，所报测试表现包含10次随机初始化；其中部分基线结果直接取自原文献，不是本轮统一重跑。TabZilla / TabRed 部分另述40次调参、ASHA 和额外划分。这些预算不能混成同一个设置，也不应把每个 λ 的训练、预训练和完整搜索成本隐藏成“仅训练100轮”。

正文强调先按验证指标调参，再评价其他划分；附录又出现“best test metrics over 40 trials”的表述，存在需要实现澄清的选模措辞。本项目明确只用训练和验证信息选配置，测试集保留为最终报告，不能照搬有歧义的表述。TabRed 的随机与时间划分应分开看再给汇总，不能用混合平均证明时间漂移已解决。

### （三）我们应先回答的三个实验问题

| **问题** | **固定什么、改变什么** | **主要读出** |
|---|---|---|
| score 是否准确描述旧模型的删除敏感度 | 固定同一个预训练旧网络、样本和基线替换方式，比较已有一阶、纯经验 Fisher 平方项、一阶加平方项及已有归因分数 | 对实际单特征删除损失的排序一致性、误差，以及同样删除数下的损失曲线 |
| gate / ADMM 是否确实改善选择 | 固定数据实例、划分、初始化和任务预算，逐项改变 score scaling、稀疏坐标或 gate 结构 | Top-k 恢复、支持集、实际模型稀疏度、预测变化及一致性残差 |
| 选择是否能服务旧网络及其他模型 | 把每种方法的排名或支持作用到同一个固定旧网络，另行评价同一骨干上的重训价值 | 旧网络不重训的删除曲线，与独立标记的重训预测结果 |

第一项先不让模型随 score 改变，可以避免“评分更准”与“训练到了另一模型”混淆。以标准化后置零为统一替换基线时，直接删除损失为：

$$\Delta\ell_{nj}=\ell\bigl(f(x_n^{j\leftarrow0}),y_n\bigr)-\ell\bigl(f(x_n),y_n\bigr).$$

这是真实计算的指定扰动响应，用于检验 Taylor / Fisher 近似，不是因果效应。样本级评价使用对应的局部 score，整体评价再沿预先指定样本集合聚合。对最终适配后模型也做同样检查，但另列模型身份。score 的绝对值若作为重要性读出，就与删除响应绝对值比较；有符号近似则与有符号响应比较，不能事后挑选较好看的符号口径。

还应同时报告两种删除方向：从最重要端删除应较快破坏模型，从最不重要端删除应较好维持模型。只报一个方向可能掩盖尾部排名错误或冗余效应。单特征响应与联合删除曲线分开计算，交互存在时不能把单特征分数之和当成联合删除损失。旧网络保持固定，后续微调或重训另列结果。

### （四）数据设计：在已有任务上增加有辨别力的压力测试

第一层继续用仓库已有 XOR、Ring、混合任务与真实支持集，保持 Top-k recovery 的既有主指标。先对齐本轮配对数据，不把历史外部 baseline 的不同数据与预算结果直接纳入新的差值分析。记录旧模型自身是否已学会任务，避免把旧模型未学会 XOR 的失败全算给解释算法。

第二层借鉴噪声注入，但区分三种条件：独立无关噪声、相关冗余副本、包含可预测变换的二阶额外特征。后两种未必是真正无关变量。需要先明确数据生成过程和真实支持，必要时同时报告原始变量定位与信息等价组覆盖，避免某方法选了等价副本却被一律判错。真实数据的“原始列对额外噪声”恢复仅是可控抗噪评价，不升级为全部原始变量的相关性真值。

第三层为候选三设计局部支持真值：不同样本区域依赖不同 feature，分别检查每条样本的恢复与整体聚合。保存每个 feature 的实际相关样本比例。整体平均可能淹没少数样本依赖的特征，因此按区域或局部支持模式分层报告，不只给一个全局排名。用于机制发现的数据实例与最终确认实例分开保留，不以同一数据上的反复调参当作独立确认。

候选二另加一个很有针对性的诊断：对 gate 与相邻权重列做保持预测不变的反向缩放，比较幅度坐标、score 和最终读出的变化。这个诊断比一味扩大数据集数量更直接检验所声称的尺度补偿问题。它只检查指定重参数化，不能外推整个网络对所有缩放不变。

### （五）四个候选应怎样消融

| **候选** | **最先需要的对照** | **需要排除的混淆** |
|---|---|---|
| 一：只改 score | 原一阶、纯平方项、一阶加平方项。可行时加不使用 score scaling 的控制，score 到系数的映射预先固定 | 排名读出变化和实际训练反馈分开记录。临近收敛与较早 checkpoint 分层比较，不能仅因平均梯度小就认定经验 Fisher 仍有效 |
| 二：有效幅度 | gate 坐标与有效幅度坐标采用同一 score 做配对，再在选定设置上检查是否与新 score 有交互 | 不同时更换 forward 归一化、骨干与优化器。除最终恢复外，记录 gate、列范数、乘积幅度和预测是否保持不变 |
| 三：逐样本 gate | 共享 gate、独立 per-sample gate，及 Localized Lasso / 已有局部归因对照 | 固定旧网络只拟合 gate，与联合调整模型分开。保存稳定 sample ID、G/Z/U 及整表 PQI 定义，不把批内独立目标冒充整表目标 |
| 四：LassoNet 到 PQI | 原版 LassoNet、ADMM-Lasso、ADMM-PQI。LassoFlexNet 作为完整方法基线，结构与求解器对照另列 | 不一步同时换惩罚和求解器。算法 2 未澄清前不自行补实现再称官方复现，明确所用版本、修正与结果身份 |

先完成单因素与必要的小范围交互，不把全部组件、数据和超参数机械做全排列。EMA、训练顺序、额外预训练等如果借鉴引入，应单独作为候选或控制，不悄悄并入“只改 score”的实验。

### （六）指标、稳定性与预算的共同口径

标准化、编码或 PLE 边界、选择器与超参数仅用允许的训练和验证信息拟合。候选三与 Localized Lasso 还要明确未见样本的局部参数如何产生：若使用测试标签拟合该样本的 gate 或局部系数，就不能将相应预测当作无标签部署下的测试泛化结果。可以研究带标签样本的局部重要性，但须保留这个评价身份。使用未标注测试特征构图的传导设置也应单列，不与纯归纳结果混用。

重要性评价保留 Top-k recovery，并按问题补充 AP / PR 曲线、恢复 AUROC、支持 FDR 与每样本宏平均指标。恢复 AP 的参照与真特征占比有关，不能跨维度直接比较一个孤立 AP。这里“恢复 AUROC”是以 feature 为对象的恢复指标，不是以样本为对象的分类 AUROC。原始有符号分数、转为非负的 importance 和 ADMM 系数分别保存。

模型侧记录保留或删除特征数、旧网络输出与损失变化，以及重训预测指标。多分类可以使用交叉熵与预先固定的分类指标，不能继续默认 target=0 的归因就代表全部类别的损失重要性。分类、回归的原始 accuracy、AUROC、RMSE 分开汇报，跨数据集仅使用预先定义的秩或相对变化，不能直接平均不同量纲的分数。

稳定性至少分开看数据变化、模型初始化变化与超参数变化。支持集合用 Jaccard 等重合指标，排名用适合并列的秩相关，同时配合恢复质量。大量同分或全部为零时，固定平局策略并明确退化情况，不能把稳定地给出错误排名当作高质量。独立数据实例是合成数据主结论的重要重复单位，不能把同一实例的多个初始化或 CV 折当成完全独立样本来扩大显著性。

预算比较同时记录试验数、搜索空间、总路径训练量、梯度或模型求值数、墙钟与峰值显存。相同 trial 数不保证相同成本，尤其是 per-sample 梯度、整表 G/Z/U 与多 λ 路径。Fisher 分数要核对真实逐样本平方平均，不能用 batch 平均梯度平方替代，并分别记录其额外成本。失败、数值异常和超时进入分母，不只报告成功运行。

固定 k 的比较用于对齐选择预算，所有方法使用相同的 k 信息。自动选稀疏度的比较则另列验证集选模规则与实际保留数，不能一边用真值 k、一边自动选择后把两者混成同一排名。L1/L2 的统一尺度不变性意味着原 Lasso 的 λ 路径与支持单调性不能直接继承，需明确 PQI 的实际选择机制与停止规则。

对 ADMM，辅助变量零数与模型真正切断的路径数分开记录。至少保存原始/对偶残差、有效支持和实际前向影响；候选四另外记录 hierarchy 违反量。不能仅用 Z 变稀疏或优化损失下降证明完成了 feature selection。

### （七）建议的实施顺序与验收问题

1. 先做固定旧网络的 score 与真实删除响应对照，验证候选一的必要性，并完成候选二的保持预测缩放诊断
2. 在既有合成任务上，采用同实例、同划分、同起点的配对实验，检查 score 与有效幅度的训练作用，保留各项失败结果
3. 加局部支持真值，单独评价候选三与 Localized Lasso，并检查局部结果如何影响整体排名
4. 实现和核对候选四的 ADMM-Lasso，再替换 PQI；将可核实版本的 LassoFlexNet 接入完整方法比较
5. 在机制与读出固定后增加真实数据抗噪和同骨干下游评价，再按实际研究目标扩展 TabZilla 或时间漂移数据

首轮的关键验收是：评分是否更接近指定旧模型的真实删除响应，效果是否在独立确认数据上保持，等稀疏预算下是否改善恢复，辅助稀疏是否兑现为实际路径删除，以及收益是否值得额外计算和存储。TabRed 全套预测榜单、基础模型大规模比较和完整编码/Mixer 组件移植不作为证明这几个问题的前置条件。

以上是从原论文提炼并针对本项目提出的实验建议，尚未执行，不改变暂停实验的当前状态。原论文的实测结果、其协议疑点与本项目待验证设想分别保留。

## 十四、主任务固定为 Feature Ranking

本项目主任务按已澄清的口径固定为：方法输出每个 feature 的连续分数或排名，不要求同时决定实际保留个数。已有 XOR、Ring 与混合任务的真实支持集用于直接评价 ranking。十二真实数据集的下游 benchmark、TabZilla 和 TabReD 作为另行标记的补充，不替代已知支持真值的排名主评价。

原 NeurIPS 2023 十二数据集 benchmark 的主要管线是“排名 → 固定预算取前 k 个 → 训练下游 MLP / FT-Transformer → 报告预测表现”。论文 §6 明确 k 为加入额外列之前的原始特征数，未要求每个算法自行发现这个个数。它也有排名与噪声区分的辅助评价，但主表的 accuracy / RMSE 反映下游预测价值。[原论文 §5–6 / F.2](https://papers.neurips.cc/paper_files/paper/2023/file/82f39c7409155b74d15d73b048f06771-Paper-Datasets_and_Benchmarks.pdf)

因此，需要区分三件事：方法是否把已知相关特征排在前面，固定排名截断后能否支持下游训练，以及方法能否自行确定保留个数。第三件事会增加阈值或选模问题，当前不作为 ranking 的必备输出。方法使用 gate 或 ADMM 并不改变这个评价约定，最终可以读出分数，而不以某次软阈值后的零数作为“正确特征数”。

主评价可以使用完整排名的 AP / 恢复 AUROC，并保留现有 recall@k。这里评测用的 k 来自真实支持集，只用于读出排名命中情况，不交给方法决定支持，也不意味着方法实际输出了 k 个非零 feature。截断排名、自动选个数和重训下游网络分别报告，不能混成同一个能力。大量分数为零或相同的情况保留平局，不用原始列顺序获得有利排名。

十二真实数据集的“原始列”只表示额外特征加入前的列，不保证每一列都是真正相关变量。二阶额外列还可能含有可用信号，原论文亦说明选中它们未必有害。因此这套数据适合研究抗额外特征干扰与下游价值，不能直接充当全部原始变量相关性已知的 ranking 真值库。

## 十五、正则强度、保留预算与下游效果

连续 importance 排名不自动给出预测最好的保留个数。固定保留 k 个、由验证集选择稀疏度、以及直接评价完整排名，是不同的输出约定。原十二数据集 benchmark 的主要下游比较预先固定 k 为原始列数，不是在证明算法能自动发现最优保留个数。

LassoNet 的原生处理是正则路径：先训练 λ=0 的稠密模型，再从较小正 λ 出发，通常按乘法逐步增大，以上一步参数 warm start 继续训练，保存不同支持数的状态。自动 lambda_start 只估计路径起点，不等于自动确定最优 λ 或最优 feature 数。M 是 hierarchy 系数，不是保留预算。[官方说明](https://github.com/lasso-net/lassonet)、[API](https://lasso-net.github.io/lassonet/api/)

普通 path 返回整条路径，需要另行选中并载入某个状态。官方 LassoNetClassifierCV / LassoNetRegressorCV 在多折上评价路径，按平均验证得分选择 best_lambda_，再在该强度对应的路径范围训练全数据模型。此时保留数是选定模型的支持数，是按预测准则作出的经验选择，不是一个脱离验证数据的最优预算保证。官方说明还区分了论文中的路径方法与后续增加的 CV、自动起点等库功能。

本次固定官方源码 e09419126808afe185b921c790802f50695a298e 中，BaseLassoNetCV 以各折得分插值后取均值最大点。当前项目 feature-selection 分支的 src/core.py 则使用普通 LassoNetClassifier，先获取 path 和 feature_importances_，再按 path 中 val_loss 最小项载入预测状态。重要性来自整条删除路径，预测状态来自验证选模，二者并非同一终点权重。不能把普通 fit 跑到路径末端当成已经载入最优稀疏模型。

对最终使用旧网络的流程，最关键的是验证目标一致。如果 λ 的支持集被拿去重新训练另一个下游网络，应该用该下游网络的验证表现选择 λ 或保留 k；LassoNet 自己的 CV 结果不能直接保证另一个模型的效果。每个候选支持都重训下游会增加成本，需要明确记账。旧网不重训的删除影响与重新训练后的预测价值也分别保存。

若外部先给预算 K，可以在路径中满足支持数不超过 K 的状态里按验证指标选，或明确采用 ranking 的 Top-K 截断。后者不自动等价于某个原生路径状态，路径也可能一次删掉多个 feature，未必恰好经过每个整数 K。若没有外部 K，可以用验证最优点，或预先指定允许的预测退化后选最小支持；这仍需要明确选择准则，不存在仅看 importance 就能保证下游足够好的通用个数公式。

因此，本项目当前主结果先报告 ranking 与已知真值的一致性；需要下游结果时，再单独固定 budget 曲线或验证集选预算的完整流程，测试集只作最终评价。不能因为这种预算选择较复杂，就把预测结果与排序质量视为同一个问题。


## 沟通与同步入口

[AI 沟通流程](https://dreamsoul.feishu.cn/docx/BXz0dmV0QoNkjyxcO6Ec6ijencm) 保留讨论和决定的演进。完整沟通原件、用户配置、工具回执及论文 PDF 不随此文档提交。
