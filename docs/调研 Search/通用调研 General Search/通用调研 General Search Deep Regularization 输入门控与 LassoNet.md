# 通用调研 General Search Deep Regularization 输入门控与 LassoNet

## 一、问题与已有依据

当前实验执行口径以 [Deep Regularization 研究计划](<../../研究计划 Research Plan Deep Regularization.md>) 为准；[飞书计划](<../../研究计划 Research Plan Deep Regularization.md>) 已同步。本稿保留方法推导与源码依据。

Deep Regularization 当前需要解释输入门控稀疏化与 LassoNet 的机制差异，尤其是控制参数、第一层权重和约束更新之间的关系。本稿整理数学结构、源码差异与四个候选方案，作为继续研究的工作稿。四个候选及 ADMM 更新从第六章进入。

源码核对显示，输入乘法 gate 与 LassoNet 都能实现输入级特征选择，但控制量和优化方式不同。`feature-selection` 分支的公开入口默认使用 Ratio Norm 与 ADMM，并有 L1 消融，不能笼统写成只在 gate 上做 Lasso。主分支保留的后续研究又包含有效 gate 方案，需要按版本区分。

本次核对固定于2026年11月27日：`main` 为 `e0626201327410f20037940658fec0ad60549273`，`feature-selection` 为 `555ba11e8872bd6618cd721bd3b08f7128db9e6b`，LassoNet 官方实现为 `e09419126808afe185b921c790802f50695a298e`。本文为源码分析与数学推导，未启动训练或复现实验。

[2026年11月20日会议纪要](https://dreamsoul.feishu.cn/docx/M4zodUdZfotn15xndUWc05oPnmg) 记载，高维与 XOR 任务仍需核查，现场消融汇报提示 hierarchy 的作用值得单独研究。这些属于会议汇报，尚不能替代可复现实验。

## 二、输入门控方案

设输入为 $x\in\mathbb R^d$，共享 gate 为 $a\in\mathbb R^d$，网络为 $g_W$：

$$z(x)=g_W(a\odot x).$$

分类时，$z(x)$ 是类别 logits。一个 gate 控制一个输入维度，该维度被切断时影响所有类别。若选用 L1 惩罚，其基本数学形式为：

$$\min_{a,W}\ \mathcal L(y,g_W(a\odot x))+\lambda\|a\|_1.$$

第一层的有效权重为 $W^{(1)}\operatorname{diag}(a)$。如果参数化允许，且第一层权重没有有效限制，对任意 $c>1$ 作如下变换：

$$a_j\leftarrow a_j/c,\qquad W_{:,j}^{(1)}\leftarrow cW_{:,j}^{(1)}.$$

有效权重和预测保持不变，gate 的 $L_1$ 惩罚却下降。这个推导针对无其他有效约束的原始 gate 惩罚。已核对的早期实现有权重衰减，后续实现还有有效 gate，因此不能将该推导直接作为当前高维失效的原因。

### （一）源码位置与版本边界

源码核对使用固定的 `main`、`dev` 与 `feature-selection` 分支版本。`main` 的 [src/models.py](https://github.com/DreamSoul-AI/Deep-Regularization/blob/e0626201327410f20037940658fec0ad60549273/src/models.py) 只有普通 MLP，并没有这里讨论的 gate。`dev` 的本次分支头为 `717f8b8158dedb90d6020e96ba12346f59344aa5`，其 `src` 与研究证据目录相对本次 `main` 没有差异。

可编辑的历史特征选择实现位于 `feature-selection` 分支的 [src/admm_input_group_wrapper.py](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py)。后续研究实现以不可变快照保留在 `main` 的 `docs/research/20260920-xor-ring-thread/sources/`，[来源索引](https://github.com/DreamSoul-AI/Deep-Regularization/blob/e0626201327410f20037940658fec0ad60549273/docs/research/20260920-xor-ring-thread/PROVENANCE.md#L176-L182) 标明原始模块身份。快照不能当作已经接入 `main/src` 的模块，也不代表2026年11月20日汇报所用的未提交代码。

### （二）原始 gate 的前向计算

`GatedFeatureSelectionMLP` 将 gate 定义为全一初始化的 $d$ 维 `nn.Parameter`。默认 `bounded_gate=False`，因此它是可正可负的直接参数，没有 sigmoid 或非负限制。前向为 `self.layers(x * self.get_gate_values(training_drop=True))`，没有输入到输出的线性 skip。[模型定义与前向](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L143-L245)

默认骨干为两层、每层32个隐藏单元，激活为 Mish。训练时 `feat_drop=0.6`，每次前向抽一个长度为 $d$ 的特征 mask，在该 batch 的所有样本间共享，并除以保留率。预测时关闭此随机 mask。二分类输出一个 logit 并使用 `BCEWithLogitsLoss`，多分类输出类别向量并使用 `log_softmax + NLLLoss`。[损失选择](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L512-L546)

可选的 `bounded_gate=True` 改用 sigmoid，但该分支的 ADMM 在 sigmoid 前的原始参数上操作。原始参数为零时，有效乘法 gate 是0.5，不能沿用“参数归零即切断输入”的解释。可选的第一层按列归一化也存在，但公开入口默认关闭。[入口装配](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L1311-L1416)

### （三）公开入口的 ADMM 更新与最终读出

`experiments/main/run_admm_input_group_benchmark.py` 调用 `run_admm_input_group`，显式开启 `use_ratio_norm=True` 和 `use_admm=True`。入口默认 $C=0.05$、共500轮、前120轮 warm-up、batch size 为64。先用训练折均值和标准差标准化输入，再建立模型并进入 `_train_input_group`。[调用入口](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/experiments/main/run_admm_input_group_benchmark.py#L97-L108)、[参数与装配](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L1352-L1424)

默认非 sigmoid 路径引入辅助变量 $z$ 和缩放对偶变量 $u$，用二次项推动 $a=z$。每个 mini-batch 更新网络和 gate 所用的损失为：

$$\mathcal L_{\mathrm{data}}+\frac{\rho}{2}\|a-z+u\|_2^2.$$

warm-up 与 ADMM 阶段都使用 Adam，`weight_decay=0.001` 作用于传入优化器的全部模型参数，包括 gate。它对前述缩放补偿有代价约束，不能说第一层可以无成本无限放大。[优化器与神经网络更新](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L788-L836)

每轮完成 mini-batch 更新后，刷新得到的第一层列范数 $s_j=\|W_{:,j}^{(1)}\|_2+10^{-8}$ 用来设置阈值系数 $\ell_j=\operatorname{clip}(C/s_j,10^{-8},0.5)$。对 $v=a+u$ 先做：

$$t_j=\operatorname{sign}(v_j)\max(|v_j|-\ell_j/\rho,0).$$

开启 Ratio Norm 时，再按代码计算全局三次方程系数，求 $\tau^3-\tau-D=0$，令 $z=\tau t$。关闭 Ratio Norm 的 L1 消融直接令 $z=t$。之后更新 $u\leftarrow u+a-z$，并按原始残差与对偶残差调整 $\rho$。这是该分支实际采用的“软阈值加三次方程缩放”，本次没有证明其等于一般 Ratio Norm 目标的精确全局近端解，不能只根据源码注释认定。[完整每轮更新](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L838-L964)

这个默认循环没有在每轮把 $z$ 直接覆盖写回 `model.gate`，因此辅助变量稀疏不等于用于预测的 gate 已经精确归零。最终 `scores` 来自 gate 的绝对值，而非辅助变量 $z$ 或第一层列范数的乘积。[特征读出](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L1151-L1172)

### （四）后续研究中的有效 gate

保留的研究实现包含 `GatedExplicitScaleMLP`。为避免与前述原始 gate 符号混淆，记原始 gate 为 $a_j$，第一层列长度为 $r_j=\|W_{:,j}^{(1)}\|_2$，有效 gate 为：

$$q_j=r_j a_j,\qquad V_{:,j}=W_{:,j}^{(1)}/r_j.$$

在非零列上，前向等价于 $V\operatorname{diag}(q)x$，且 $\|V_{:,j}\|_2=1$。代码使用安全分母处理零列，约束和读出仍使用真实列范数。`get_constraint_coordinate()` 返回 $q$，`get_feature_selection_scores()` 返回 $|q|$，`get_feature_scores()` 返回全一以关闭旧的 $C/s_j$ 阈值加权。[有效 gate 模型](https://github.com/DreamSoul-AI/Deep-Regularization/blob/e0626201327410f20037940658fec0ad60549273/docs/research/20260920-xor-ring-thread/sources/f52bf05d1773d87588b3.py#L770-L892)

这一版本将一致性条件改成 $z=q=r\odot a$。二次项通过 $r(W)$ 反传到第一层权重，神经网络更新后重新计算列长度，再形成辅助变量更新的中心。[可微约束和更新顺序](https://github.com/DreamSoul-AI/Deep-Regularization/blob/e0626201327410f20037940658fec0ad60549273/docs/research/20260920-xor-ring-thread/sources/f52bf05d1773d87588b3.py#L3647-L3694)

因此，执行 $a_j\leftarrow a_j/c$、$W_{:,j}^{(1)}\leftarrow cW_{:,j}^{(1)}$ 时，$q_j$ 保持不变，无法再通过这个缩放降低有效 gate 的惩罚。这解决的是原始因子读出的尺度歧义，不构成解决全部深层缩放自由度或高维失效的证明。

已有 [C-path 诊断调用](https://github.com/DreamSoul-AI/Deep-Regularization/blob/e0626201327410f20037940658fec0ad60549273/docs/research/20260920-xor-ring-thread/sources/c948bea9ba04f4fc1da8.py#L354-L405) 明确选用该模型及 `canonical_signed` Ratio Norm 更新，关闭旧阈值加权，并设置 `write_z_to_gate=False`。快照中同时存在原始 gate、冻结列长度加权、可微有效 gate、父特征共享 gate 等路径，必须按具体入口配置解释，不能合并成一个“当前 gate 算法”。

## 三、LassoNet 的结构

### （一）线性分支与维度

单输出回归使用 $f(x)=\theta^\top x+g_W(x)$，其中 $\theta\in\mathbb R^d$，两条分支都输出标量。$\theta^\top x$ 是随输入变化的线性预测，不是固定 bias。

多分类使用 $C$ 维 logits：

$$z(x)=\Theta x+g_W(x),\qquad\Theta\in\mathbb R^{C\times d}.$$

两条分支在 softmax 之前相加。采用第一层权重形状 $K\times d$ 的约定，$W_{:,j}^{(1)}$ 是输入 $j$ 通往全部 $K$ 个隐藏神经元的连接，$\Theta_{:,j}$ 是该输入通往全部类别的线性连接。

### （二）稀疏惩罚与层级约束

单输出的约束为 $\|W_{:,j}^{(1)}\|_\infty\le M|\theta_j|$。多输出的官方实现按输入列使用组范数，对应形式为：

$$\min_{\Theta,W}\ \mathcal L(y,z(x))+\lambda\sum_j\|\Theta_{:,j}\|_2\quad\text{subject to}\quad\|W_{:,j}^{(1)}\|_\infty\le M\|\Theta_{:,j}\|_2,\quad\forall j.$$

这里 subject to 表示“满足以下约束”。$\lambda$ 控制正则强度，$M$ 是控制约束松紧的超参数，二者需要选择。$M$ 不由当前权重唯一计算得到。当某列 $\Theta_{:,j}$ 归零时，对应第一层连接必须全部归零，该输入被整体移除。

### （三）约束怎样执行

训练先对预测损失进行普通梯度更新，再执行联合近端更新。对单个输入，记梯度更新结果为 $\bar v=\bar\Theta_{:,j}$、$\bar u=\bar W_{:,j}^{(1)}$，步长为 $\eta$，基本近端子问题为：

$$\min_{v,u}\ \frac12\|v-\bar v\|_2^2+\frac12\|u-\bar u\|_2^2+\eta\lambda\|v\|_2\quad\text{subject to}\quad\|u\|_\infty\le M\|v\|_2.$$

该操作联合调整两个分支，而非固定线性系数后只裁剪第一层。作者给出了基于排序和阈值候选的近端求解算法，每次完成近端更新后满足约束，无需等整个训练结束。这里采用原始的欧氏近端与梯度更新形式，其他优化器的收敛性质需要另行判断。

该子问题的可行集闭且非空，零参数可行。二次距离项使目标在参数范数趋于无穷时趋于无穷，因此最小值能够取得。原方法的近端子问题可求全局解，整个神经网络训练仍为非凸优化，不由此获得全局最优保证。

## 四、改用 L1/L2 的含义

对原始 gate 可使用 $R(a)=\|a\|_1/\|a\|_2$，对有效 gate 则可使用 $R(q)$。Ratio Norm 在已核对的分支和研究快照中都已有实现，所以相对这些实现，它不是全新的候选改法。将 LassoNet 的控制参数惩罚改成比值仍是另一项方法修改。PQI 的具体定义、符号方向及归一化应按所选配置确认，不能把所有 PQI 写法视为未经缩放的同一个公式。

比值对整个向量的统一非零缩放保持不变，因此“整体缩小 gate 降低惩罚”的论证不能直接照搬到纯比值惩罚。但各坐标仍可能与对应第一层权重互相补偿，尺度与特征重要性的对应关系仍需检查。

比值的分母耦合各坐标，原始 LassoNet 按特征分解的近端求解不能直接照用。还需明确零向量处的定义，使用 $\epsilon$ 等平滑方式会改变精确的尺度性质。多分类时应先决定惩罚矩阵元素还是每个输入列的范数，后者才直接对应输入级选择。

## 五、与 LassoNet 的实现差异

| **比较项** | **原始 gate 分支默认路径** | **有效 gate 研究路径** | **LassoNet 官方路径** |
| --- | --- | --- | --- |
| 前向控制 | $g_W(a\odot x)$ | $g_{V,\ldots}(q\odot x)$ | $\Theta x+g_W(x)$ |
| 控制量 | 每个输入一个标量 | 每个输入一个有效标量 | 每个输入一列线性系数 |
| 默认稀疏处理 | 自适应阈值与 Ratio Norm 缩放 | 所选配置上的 Ratio Norm 更新 | 线性分支的组 L1 近端 |
| 连接约束 | ADMM 推动 $a=z$ | ADMM 推动 $q=z$ | 第一层最大绝对权重受线性列范数限制 |
| 约束满足方式 | 迭代跟踪一致性残差 | 迭代跟踪一致性残差 | 每次近端更新后直接满足不等式 |
| 稀疏操作频率 | 每轮一次辅助变量更新 | 所核对调用每轮一次 | 每个优化器更新后一次联合近端 |
| 特征读出 | 最终 $\lvert a_j\rvert$ | 所选配置的 $\lvert q_j\rvert$ 或路径读出 | 非零列掩码及路径退出强度 |
| 主要系数 | $C,\rho$ 及阈值加权设置 | $C,\rho$ 与更新模式 | $\lambda,M$ |

LassoNet 的 [训练接口](https://github.com/lasso-net/lassonet/blob/e09419126808afe185b921c790802f50695a298e/lassonet/interfaces.py#L294-L324) 在每次 `optimizer.step` 后调用 `model.prox(lambda_=lambda*lr, M=M)`。[模型装配](https://github.com/lasso-net/lassonet/blob/e09419126808afe185b921c790802f50695a298e/lassonet/model.py#L34-L70) 将 skip 和第一层一起传给近端算子，[近端实现](https://github.com/lasso-net/lassonet/blob/e09419126808afe185b921c790802f50695a298e/lassonet/prox.py#L26-L55) 联合缩放 skip 列并截断第一层连接。因此不能把它的 hierarchy 理解成 gate 版本中 $C/s_j$ 的加权阈值，也不能把 gate 版本的 $\rho$ 当成 LassoNet 的 $M$。

LassoNet 的线性列必须参与预测，同时作为第一层连接幅度的控制量。有效 gate 方案则直接控制第一层有效输入列的幅度，没有这条额外线性输出分支。两者共享输入级稀疏化目标，但并非换一个惩罚公式就互相等价。上述结构和更新差异能够由源码确认，哪一项导致已有 XOR / Ring 性能差距仍需要独立对照。

## 六、四个候选的分工与共用 ADMM 约定

### （一）候选总览

前三个候选沿原有 gate 与 PQI／ADMM 方案展开，第四个独立研究 LassoNet 的 ADMM 与 PQI 变种。第四个先建立保留原 Lasso 惩罚和 hierarchy 的 ADMM 基线，再替换为 PQI，不与前三个候选混为同一结构。以下统一用 $f_W$ 表示预测模型，$g$ 表示共享 gate，$G$ 表示逐样本 gate，$S$ 表示梯度产生的 score，$Z$ 表示稀疏辅助变量，$U$ 表示缩放对偶变量。原始数据 $X$ 在四个候选中均保持固定。本节后续的 score scaling 和通用更新说明主要服务于前三个候选，候选四的独立分裂与更新见第十章。

| **候选** | **主要改动** | **ADMM 跟踪量** | **重要性读出** |
| --- | --- | --- | --- |
| 一 评分改动 | 原方案的 score 加入一阶和曲率信息 | 原方案选定的共享 gate 坐标 | feature 级 score |
| 二 有效幅度 | 普通前向，显式计算 gate 与权重列范数的乘积 | $q_j=g_j\lVert W_{:,j}^{(1)}\rVert_2$ | score，另记录有效幅度 |
| 三 逐样本 gate | 为每个样本、每个 feature 保存独立 gate | $G\in\mathbb R^{N\times d}$ | 局部 score 及跨样本求和的全局 score |
| 四 PQI-LassoNet | 先做 LassoNet-ADMM，再替换 PQI | 线性分支及第一层的分裂副本 | 沿用基线的特征读出，另行对齐比较 |

候选一改变评分来源，候选二改变稀疏约束坐标，候选三改变 gate 的粒度。前三者可以分别比较，也可以明确组合，但不能将多个改动的组合收益归因于其中单项。当前均为候选设计，尚未修改模型代码或启动训练。

### （二）score 的职责是 scaling

score 用于控制稀疏更新的相对强度，不把 score 本身当作新的可学习参数，也不优化 PQI(score)。每轮评分由预测损失的梯度产生，再执行 stop-gradient，转换成非负的缩放系数 $c$，在该轮稀疏更新中固定。正常预测损失仍向模型及 gate 反传，因此无需为了更新这些系数而对 score 再次求导。

有符号的删除代价与用于阈值的非负系数分开保存。符号处理、是否取绝对值、归一化和缩放映射尚待选定。一个可比较的例子是先取得非负重要性 $I_j$，再使用有界的 $c_j=\operatorname{clip}(1/(I_j+\epsilon),c_{\min},c_{\max})$，使重要 feature 少收缩。这个映射是建议选项，不是本轮已确认或已验证的默认值。不能把带负号的原始 score 直接作为 L1 阈值。

### （三）共用 ADMM 三步

记当前候选的模型及 gate 参数为 $\theta$，稀疏坐标为 $A(\theta)$，本轮固定系数为 $c^k$。一个用于说明变量关系的分裂目标是：

$$\min_{\theta,Z}\ \mathcal L(\theta)+\lambda R_{c^k}(Z)\quad\text{subject to}\quad Z=A(\theta).$$

每个外层更新包含以下步骤：

1. 更新模型及 gate，近似求解 $\theta^{k+1}\approx\arg\min_\theta\{\mathcal L(\theta)+\tfrac\rho2\|A(\theta)-Z^k+U^k\|^2\}$。向量使用 L2 范数，矩阵使用 Frobenius 范数。预测损失与一致性项共同提供更新信号
2. 重算当前坐标 $A^{k+1}$，更新 $Z^{k+1}\in\arg\min_Z\{\lambda R_{c^k}(Z)+\tfrac\rho2\|Z-(A^{k+1}+U^k)\|^2\}$
3. 更新 $U^{k+1}=U^k+A^{k+1}-Z^{k+1}$，记录原始残差 $\|A-Z\|$ 与对偶残差 $\rho\|Z^{k+1}-Z^k\|$

若将 scoring 用作迭代重加权、每轮刷新 $c^k$，目标也随之变化。这是需验证的自适应流程，不据上述写法声称固定目标 ADMM 的收敛保证。神经网络块使用有限步随机梯度，本身也是近似子问题求解。

### （四）L1 示例与 PQI 的对应位置

L1 情形可明确写成 $R_c(Z)=\sum_i c_i|Z_i|$，此时中心 $V=A^{k+1}+U^k$ 的更新为：

$$Z_i^{k+1}=\operatorname{sign}(V_i)\max(|V_i|-\lambda c_i/\rho,0).$$

PQI／Ratio Norm 替换的是第二步的稀疏更新，score 仍是 scaling。未加权的 $\|\operatorname{vec}Z\|_1/\|\operatorname{vec}Z\|_2$ 会耦合所有元素，不能用逐元素 L1 软阈值直接冒充其精确近端解。既有实现包含历史的“阈值加三次方程缩放”和后续其他求解路径，实施时须固定所选版本，以及 $c$ 进入阈值或比值项的具体方式。不同加权形式对应不同目标，不在本稿中混为一个已确定的 PQI 公式。

普通 group norm 的根号来自其惩罚定义，不是 OBD score 的要求。前三个候选均不因跨样本聚合 score 而自动引入 Group Lasso。第四个在第一阶段保留 LassoNet 已有的多分类列组范数。

## 七、候选一 原方案只修改 score

### （一）前向与评分

保持共享 gate 的前向 $h_n=x_n\odot g$、$\hat y_n=f_W(h_n)$。第一候选只替换现有稀疏更新所用的 score，不主动改变骨干、gate 参数化、PQI／ADMM 求解器或一致性坐标，也不引入 LassoNet 的线性分支。

对平均预测损失 $\mathcal L=N^{-1}\sum_n\ell_n$，删除 gate $g_j$ 的扰动是 $\Delta g=-g_je_j$。二阶 Taylor 展开为：

$$\Delta\mathcal L_j\approx-g_j\frac{\partial\mathcal L}{\partial g_j}+\frac12g_j^2(H_g)_{jj}.$$

以逐样本梯度外积近似曲率，先计算逐位置贡献：

$$t_{n,j}=g_j\frac{\partial\ell_n}{\partial g_j}=h_{n,j}\frac{\partial\ell_n}{\partial h_{n,j}},\qquad s_{n,j}=-t_{n,j}+\frac12t_{n,j}^2.$$

再跨样本聚合成共享 feature score：

$$S_j=\frac1N\sum_n s_{n,j}=-\frac1N\sum_n t_{n,j}+\frac1{2N}\sum_n t_{n,j}^2.$$

平方必须在跨样本聚合之前执行。平均梯度因正负抵消接近零时，平方平均仍可有信号；如果每个样本梯度都接近零，empirical Fisher 项也会退化。它不是始终准确的真实 Hessian。单 gate 删除只涉及一个坐标，多个 gate 联合删除仍可能包含交叉项。

### （二）ADMM 怎样接入

使用 $c^k=\operatorname{stopgrad}(\phi(S^k))$ 替换旧的 scaling，其他更新沿既定方案保持一致。若原方案在共享原始 gate 上分裂，则 $A(\theta)=g$、$Z,U\in\mathbb R^d$，模型块的损失为 $\mathcal L+\tfrac\rho2\|g-Z+U\|_2^2$。接着做原有 PQI 稀疏更新和对偶更新。若实际基线已经跟踪有效 gate，就保留其坐标，不因“只改 score”而切回原始 gate。

score 不反传进稀疏系数，因此这一候选不要求高阶自动微分。对样本互不耦合的模型，可以对 batch 预测损失之和求 activation 梯度，直接保留 batch×features 的 $t$。评分时固定 dropout 等随机因素，并处理 BatchNorm 的样本耦合。若使用 batch 平均损失，需还原梯度缩放，避免 batch 大小改变一阶项与平方项的相对权重。

### （三）最终读出与待定口径

特征重要性按聚合后的 score 排序，gate 数值与辅助变量支持集作为优化状态另存。重要性使用有符号删除增量、整体绝对值还是逐样本绝对值仍待固定。$|\mathbb E[s]|$ 与 $\mathbb E[|s|]$ 不等价，不能混用。评分梯度以预测损失为定义，不把 ADMM 增广项或正则梯度自动并入 importance。

[Taylor pruning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf) 提供梯度乘参数的评分背景。[Repair Fisher Score 的公开推导](https://arxiv.org/html/2610.30552v1#S6.SS2) 从预测分布 KL 得到二次项，KL 在零扰动处的一阶为零。本候选展开实际标签上的预测损失，保留一阶项，二者目标不同。[Empirical Fisher 的局限](https://arxiv.org/abs/1905.12558) 用于界定梯度外积与真实曲率的差异。

## 八、候选二 有效幅度与联合剪枝

### （一）普通前向与有效幅度

保留普通的第一层前向 $W^{(1)}(g\odot x)$，不在前向归一化权重。显式计算：

$$r_j=\|W_{:,j}^{(1)}\|_2,\qquad q_j=g_jr_j,\qquad\|g_jW_{:,j}^{(1)}\|_2=|q_j|.$$

“抽出 norm”指计算有效幅度 $q$，不把 $r$ 变成新的独立可训练参数。缩小 $g_j$、同比放大对应权重列时，$q_j$ 不变，因此原始 gate 与第一层之间的因子缩放不能降低 $q$ 的惩罚。深层缩放自由度仍需按实际网络另行判断。

### （二）ADMM 同时影响 gate 和权重

选择 $A(\theta)=q(g,W)$，$Z,U\in\mathbb R^d$，对应目标为 $\mathcal L+\lambda R_c(Z)$，约束 $Z=q(g,W)$。模型块使用：

$$\mathcal L+\frac\rho2\|g\odot r(W)-Z+U\|_2^2.$$

一致性项通过乘积向 gate 和第一层权重两边反传。$r$ 参与联合目标时不能 detach 后仍声称优化的是同一个目标。模型块更新后重新计算 $q$，再以 $q+U$ 为中心做 PQI／L1 稀疏更新，最后执行 $U\leftarrow U+q-Z$。

这里 $Z$ 是目标乘积，不是把 gate 和 norm 分别赋值成 $Z$。范数是权重的派生量，改变它必须通过更新对应权重列实现。分别惩罚 $g$ 与 $W$ 也不等于惩罚它们的有效乘积。

候选一的 score scaling 可以与本候选组合，但组合比较应单独标记。第二章保留研究路径已有相近的有效 gate 坐标，实施前要对齐它与本次联合清零规则的实际差异。

### （三）确定删除后两处同时清零

辅助变量 $Z_j=0$ 不代表模型中的 $q_j$ 已立即为零。按固定的删除规则确定 feature $j$ 要永久退出后，同时执行 $g_j\leftarrow0$ 与 $W_{:,j}^{(1)}\leftarrow0$，并处理梯度及优化器动量等状态，保持已删除项为零。这是明确的硬剪枝步骤，不自动等于软阈值子问题的精确解。

实际压缩时同步删除输入列 $x_j$、gate 元素和第一层对应输入权重列。第一层由 $K\times d$ 变为 $K\times(d-1)$，隐藏输出宽度 $K$ 不变。若源码按输入×隐藏维度存储，所对应的是一行，判断以真实张量形状为准。

### （四）为何暂不采用前向 weight normalization

若改成 $w_j=g_j\hat v_j$、$\hat v_j=v_j/\|v_j\|_2$，归一化会参与反传。令 $d_j=\partial\mathcal L/\partial w_j$，则：

$$\frac{\partial\mathcal L}{\partial g_j}=\hat v_j^\top d_j,\qquad\frac{\partial\mathcal L}{\partial v_j}=\frac{g_j}{\|v_j\|_2}(I-\hat v_j\hat v_j^\top)d_j.$$

原始方向参数的范数虽然不改变前向，却会影响有效梯度与步长，优化器状态还会影响其动态。本候选因此保持原前向，只将 norm 用于有效幅度坐标，不主动引入该重参数化。增加有效幅度惩罚仍会有意改变损失梯度，不能说所有优化动态完全不变。

## 九、候选三 逐样本 gate 与局部重要性

### （一）参数与研究目标

为每个样本、每个 feature 保存独立可学习 gate：

$$G\in\mathbb R^{N\times d},\qquad h_n=x_n\odot G_{n,:},\qquad\hat y_n=f_W(h_n).$$

网络参数 $W$ 在样本间共享，gate 按样本独立。预测损失更新 $W$ 和该样本的 gate 行，PQI／ADMM 推动 gate 稀疏。本候选的主要评价对象是每个样本的 feature importance，以及聚合后的整体排名。可学习 gate 的自由度为 $Nd$，本轮不以泛化到新样本为主要验收目标，不仅因自由度或过拟合疑虑否定该候选。

本方案不优化原始 $X$，也不把固定数据的 L1 当作模型正则。它用可学习的 $G$ 承接预测损失和稀疏更新两股作用。独立优化 score、直接优化数据，以及临时 batch mask 都不是本轮最终确认的第三候选。

### （二）矩阵 gate 的 ADMM

使用 $A(\theta)=G$、$Z,U\in\mathbb R^{N\times d}$。本轮按整个矩阵展平处理稀疏项，不显式按 feature 或样本引入 Group Lasso：

$$\min_{W,G,Z}\ \frac1N\sum_n\ell_n(f_W(x_n\odot G_{n,:}),y_n)+\lambda R_C(Z)\quad\text{subject to}\quad Z=G.$$

其中 $C\in\mathbb R_{\ge0}^{N\times d}$ 为 score 产生并固定的 scaling。L1 示例为 $R_C(Z)=\sum_{n,j}C_{n,j}|Z_{n,j}|$。若采用未加权 Ratio Norm，稀疏项为 $\|\operatorname{vec}Z\|_1/\|\operatorname{vec}Z\|_2$；scaling 如何进入现有 PQI 更新按第六章固定。

每轮先更新网络和 gate：

$$\mathcal L(W,G)+\frac\rho2\|G-Z+U\|_F^2.$$

再更新全部稀疏辅助变量并回写对偶状态。L1 时可写成：

$$Z_{n,j}^{k+1}=\operatorname{soft}(G_{n,j}^{k+1}+U_{n,j}^{k},\lambda C_{n,j}^k/\rho),\qquad U^{k+1}=U^k+G^{k+1}-Z^{k+1}.$$

PQI 若作用于整个矩阵，其分母耦合不同样本与 feature，需要按完整矩阵或等价的全局统计执行对应更新，不能悄悄改成每个 batch 单独做一次 PQI，并称为同一目标。可按整轮刷新 score、执行一次全局稀疏更新来定义起步版本，具体频率尚待实施时固定。

### （三）按样本身份保存状态

DataLoader 应返回稳定 sample ID，用 ID 索引 $G$、$Z$、$U$ 的行。打乱顺序、改变 batch 大小或重新分批时，同一条数据仍取回自己的状态，而不是使用 batch 内第几个位置的 gate。检查点同时保存样本映射、模型、gate、辅助变量、对偶变量及优化器状态。

mini-batch 预测损失只为当前样本行产生 gate 的任务梯度。增广项采用全矩阵求和还是均值必须与全局目标一致，随机采样子块时需要相应的样本数校正。否则改变 batch size 或 $N$ 就会改变任务项与稀疏一致性项的相对强度，不能默认为同一组 $\lambda,\rho$ 可直接复用。

### （四）逐样本评分和整体排名

对每条样本单独计算预测损失梯度：

$$t_{n,j}=G_{n,j}\frac{\partial\ell_n}{\partial G_{n,j}}=h_{n,j}\frac{\partial\ell_n}{\partial h_{n,j}},\qquad s_{n,j}=-t_{n,j}+\frac12t_{n,j}^2.$$

同一套公式产生 $N\times d$ 的局部 score。对每行 $s_{n,:}$ 排序，得到对应样本的 feature importance；跨样本求和得到全局读出：

$$S_j^{\mathrm{global}}=\sum_{n=1}^N s_{n,j}.$$

如果最终采用绝对值口径，须在局部与全局两级明确是先取绝对值再求和，还是先求和再取绝对值。用户确认的是“局部 score 读出，再 sum 得整体排名”，不是以 gate 值、优化后数据值或 $Z$ 的数值直接替代 score。局部、全局原始 score 和最终用于排名的量均应保留。

### （五）局部稀疏与整列删除的区别

$G_{n,j}=0$ 只切断样本 $n$ 的 feature $j$。其他样本仍可能使用同一 feature，因此不能因为某一行 gate 为零，就清零共享的 $W_{:,j}^{(1)}$，也不能删除全局输入列。若最后需要按全局排名选出统一特征集合，再单独执行全局选择与压缩。

$Z_{n,j}=0$ 与预测路径里的 $G_{n,j}=0$ 仍需区分。若仅通过 ADMM 一致性推动，就记录残差；若选用硬写回，必须明确写回时间和状态处理。这里不另加独立可学习 mask。

## 十、候选四 LassoNet-ADMM 到 PQI-LassoNet

### （一）两阶段顺序

本候选暂称 PQI-LassoNet。2026年11月27日确认的顺序为：先把 LassoNet 的原有稀疏目标改成 ADMM 求解并建立对照，再将 Lasso 惩罚替换为 PQI。第一阶段保留原前向和 hierarchy，第二阶段保留已核对的分裂方式，仅改变稀疏子问题，以区分求解器变化与惩罚变化的影响。

这条路线选择 ADMM 来处理 PQI 的分裂更新，不把“PQI 在数学上只能用 ADMM 求解”当作结论。原版 LassoNet 本身用联合近端更新；下面是候选的 ADMM 设计，尚未在仓库实施。

### （二）保留原结构与目标

沿用第三章的前向 $f(x)=Bx+g_W(x)$，这里将线性 skip 权重记为 $B\in\mathbb R^{C\times d}$，第一层为 $W^{(1)}\in\mathbb R^{K\times d}$。单输出时 $C=1$。第一阶段的目标保持：

$$\min_{B,W}\ \mathcal L(B,W)+\lambda\sum_j\|B_{:,j}\|_2\quad\text{subject to}\quad\|W_{:,j}^{(1)}\|_\infty\le M\|B_{:,j}\|_2.$$

单输出的列范数就是绝对值，多分类则保持原有组 L1 形式。此阶段不改成额外乘法 gate，不取消 hierarchy，不混入前三个候选的 score scaling。

### （三）第一阶段 ADMM 分裂与更新

一种便于分开稀疏项与 hierarchy 的设计，是为 skip 权重引入稀疏副本 $Z$，并为层级约束引入副本对 $(P,T)$：

$$Z=B,\qquad P=B,\qquad T=W^{(1)},\qquad\mathcal C_M=\{(P,T):\|T_{:,j}\|_\infty\le M\|P_{:,j}\|_2,\ \forall j\}.$$

分裂目标为 $\mathcal L(B,W)+\lambda R(Z)+I_{\mathcal C_M}(P,T)$，其中 $I_{\mathcal C_M}$ 在满足 hierarchy 时为零，否则为无穷。三个相等约束分别维护缩放对偶变量 $U_Z,U_P,U_T$。为简洁起见，以下用共同的 $\rho$，实际若分设罚参数需同步修改对应更新。

模型块更新 $B,W$，近似最小化：

$$\mathcal L(B,W)+\frac\rho2\bigl(\|B-Z+U_Z\|_F^2+\|B-P+U_P\|_F^2+\|W^{(1)}-T+U_T\|_F^2\bigr).$$

辅助变量块包括两个相互分离的子问题。首先对稀疏副本更新：

$$V=B^{k+1}+U_Z^k,\qquad Z^{k+1}=\operatorname{prox}_{(\lambda/\rho)R}(V).$$

原 Lasso 惩罚 $R(Z)=\sum_j\|Z_{:,j}\|_2$ 的每列更新为：

$$Z_{:,j}^{k+1}=\begin{cases}0,&\|V_{:,j}\|_2\le\lambda/\rho,\\(1-\lambda/(\rho\|V_{:,j}\|_2))V_{:,j},&\|V_{:,j}\|_2>\lambda/\rho.\end{cases}$$

其次，将 $(B^{k+1}+U_P^k,W^{(1),k+1}+U_T^k)$ 投影到 $\mathcal C_M$，得到 $(P^{k+1},T^{k+1})$。投影必须联合处理两个副本，不能只单边截断第一层后声称求得联合投影。它可按输入列拆分为低维子问题，但集合因列范数下界关系而非凸，不能直接当成普通凸锥投影。

最后更新三个对偶变量：

$$U_Z\leftarrow U_Z+B-Z,\qquad U_P\leftarrow U_P+B-P,\qquad U_T\leftarrow U_T+W^{(1)}-T.$$

$(P,T)$ 在完成投影后满足 hierarchy，实际预测使用的 $(B,W^{(1)})$ 则依赖一致性残差接近该副本。应同时记录三个一致性残差与预测模型的 hierarchy 违反量，不能把“约束副本满足”写成“实际模型每步已经满足”。这也与原版每次近端更新后直接约束预测参数的行为不同。

### （四）第一阶段的核验范围

与固定版本原版 LassoNet 对齐数据划分、骨干、初始化、损失、$M$、稀疏系数及训练预算口径，比较预测表现、特征支持集及排名、hierarchy 违反量和一致性残差。ADMM 不保证走出与原联合近端相同的训练轨迹，不以参数逐项相同作为默认验收条件，也不只用代码能运行代替目标与行为核验。

原版可作为独立参照，不需要先修改原版训练。先完成 ADMM-Lasso 的目标和更新核验，再进入 PQI，避免求解器与惩罚同时变化后无法解释结果。

### （五）第二阶段替换为 PQI

在保留模型块、hierarchy 投影和对偶更新的基础上，替换 $Z$ 的稀疏项。一个与多分类输入级选择相符、待核对具体 PQI 定义的表达是先取列幅度 $s_j(Z)=\|Z_{:,j}\|_2$，再使用：

$$R_{\mathrm{ratio}}(Z)=\frac{\sum_j s_j(Z)}{\sqrt{\sum_j s_j(Z)^2}}.$$

单输出时对应线性系数向量的 L1/L2。多分类时比值作用于列幅度向量，而非未经说明地对所有类别参数逐元素重排。此处沿用的是前面讨论的分组 PQI 候选，最终指数、维度归一化及零向量定义实施前固定。

第二阶段只将 $Z$ 子问题改成：

$$Z^{k+1}\in\arg\min_Z\left\{\lambda R_{\mathrm{PQI}}(Z)+\frac\rho2\|Z-V\|_F^2\right\}.$$

这个更新耦合各 feature，不能继续逐列照用第一阶段的 group soft-thresholding。ADMM 将其从 hierarchy 子问题中隔离出来，但不会自动给出 PQI 的精确解，仍需匹配该比值定义的求解器。由于 PQI 不直接惩罚统一幅度，原 Lasso 稀疏路径的系数、步长及退出条件也不能未经核验照搬。

### （六）读出与实施边界

先保持第一阶段与原版一致的特征读出策略，另保存 $B$ 与 $Z$ 的列幅度、辅助支持集、实际 hierarchy 违反量。第二阶段再比较同一读出口径下的 feature 排名和预测表现，不能仅因 $Z$ 更稀疏就认定实际模型已停止使用对应输入。

本候选是“ADMM-Lasso 基线 → PQI 替换”的研究设计，尚未运行代码或训练。它与前三个候选独立，后续若叠加一阶加曲率的 score scaling，应单列组合，不自动并入本候选的基本版。

## 十一、实施前待固定的事项

1. 确认实际接续的分支、提交与配置，区分历史 `feature-selection`、保留研究快照和11月汇报实现
2. 固定 score 的符号或绝对值口径、跨样本求和范围、scaling 映射与刷新频率，分别保留原始 score、非负 importance 和稀疏系数
3. 固定 PQI 的实际公式、零点定义、维度归一化及 scaling 接入位置，核对目标与求解步骤一致性
4. 对齐 ADMM 稀疏坐标、增广项归一化、状态持久化、辅助变量写回和硬剪枝规则，避免用 $Z$ 的零直接冒充预测路径已删除
5. 候选四先核验 ADMM-Lasso 与原版的目标及约束处理，再替换 PQI，分别保存两阶段依据
6. 按单项候选及明确组合比较局部 importance、聚合排名与指定删除操作的预测损失变化。模型可运行、评分符合推导和重要性评价有效分别验收

7. 将 LassoFlexNet（ICML 2026）与 Localized Lasso（AISTATS 2017）列为正式 baseline，分别对照整体与局部特征选择。实现状态、整体聚合规则和旧网络评价口径见 [基线清单](<../文献调研 Literature Search/文献调研 Literature Search Deep Regularization.md#十正式基线与评价口径>)

8. 实验分成两块正式评价：A 沿用有真实支持的原 benchmark 评价 feature ranking；B 按 LassoFlexNet 的十二真实数据集协议评价同预算下的下游效果。TabZilla 与 TabReD 留作扩展。数据集、baseline、正则参数选择与实施依赖统一见 [Deep Regularization 研究计划](<../../研究计划 Research Plan Deep Regularization.md>)。

以上为研究候选整理，不代表已授权恢复暂停实验或已完成性能验证。第四个候选单独采用 LassoNet 结构，不作为前三个 gate 候选的额外结构要求。

## 十二、资料与仓库入口

- [四个候选相近文献](<../文献调研 Literature Search/文献调研 Literature Search Deep Regularization.md>)：2026年11月27日定向检索。候选一对照 Taylor / Group Fisher 与加权 ADMM，候选二对照输入列和尺度稀疏，候选三对照独立局部参数的 Localized Lasso 与共享 selector 的 LSPIN，候选四对照 LassoNet、L1/L2-ADMM 及 LassoFlexNet。具体组合是否构成贡献仍需进一步比较，未检到同名方法不等于没有先例
- [Deep-Regularization 仓库](https://github.com/DreamSoul-AI/Deep-Regularization)
- [LassoNet 论文](https://jmlr.org/papers/v22/20-848.html)
- [LassoNet 方法正文与近端算法](https://pmc.ncbi.nlm.nih.gov/articles/PMC9453696/)
- [官方模型实现](https://github.com/lasso-net/lassonet/blob/e09419126808afe185b921c790802f50695a298e/lassonet/model.py)
- [官方近端实现](https://github.com/lasso-net/lassonet/blob/e09419126808afe185b921c790802f50695a298e/lassonet/prox.py)

仓库核对日期为2026年11月27日。`main` 最近提交是北京时间2026年11月19日的 [Workspace 规则采用](https://github.com/DreamSoul-AI/Deep-Regularization/commit/e0626201327410f20037940658fec0ad60549273)，涉及规则与模板。`dev` 在 [11月26日](https://github.com/DreamSoul-AI/Deep-Regularization/commit/717f8b8158dedb90d6020e96ba12346f59344aa5) 还有规则维护，因此11月19日不能表述为全仓库所有分支的最近更新。`main` 往前的研究资料更新是 [2026年9月20日 XOR / Ring 总结](https://github.com/DreamSoul-AI/Deep-Regularization/commit/da0988e0d0cd5a5404218172364f34f0f9bdfb87)。README 中实验自9月18日暂停的表述属于仓库记录，不能覆盖11月会议或代表所有分支、未提交实验和当前作业状态。

本次完成前向、入口装配、损失、ADMM 辅助变量更新及读出的静态追踪。未安装训练依赖、未执行训练或推理、未修改模型代码，未将分支或快照提升为默认版本。方法链接均固定到所读提交；LassoNet 官方当前实现未必与历史基准运行时的安装版本完全相同。


## 沟通与同步入口

[AI 沟通流程](https://dreamsoul.feishu.cn/docx/BXz0dmV0QoNkjyxcO6Ec6ijencm) 保留讨论和决定的演进。完整沟通原件、用户配置、工具回执及论文 PDF 不随此文档提交。
