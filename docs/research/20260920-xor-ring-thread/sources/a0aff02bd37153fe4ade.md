# R67：两项经验搭配的共同 XOR/Ring 筛选

日期：2026-09-17。96/96条正式训练完成，零正式失败/重试；独立审计通过96份checkpoint、48份复用参照、8份独立重构数据及192组训练/测试预测。

没有配方达到预先固定的共同改进筛选规则。独立确认未触发，整体目标仍未解决。

## 最终 Top2

每格两实例×六折，真实坐标分母24，完整恢复折分母12。既有Five参照直接复用；不新增Ring baseline训练。

| 任务 | 维度 | 已有Five 坐标 / 完整折 | SlowPath 坐标 / 完整折 | EarlySlowPath 坐标 / 完整折 |
|---|---:|---:|---:|---:|
| xor | 256 | 2/24 · 0/12 | 3/24 · 0/12 | 0/24 · 0/12 |
| xor | 512 | 1/24 · 0/12 | 0/24 · 0/12 | 0/24 · 0/12 |
| ring | 256 | 18/24 · 6/12 | 17/24 · 5/12 | 15/24 · 3/12 |
| ring | 512 | 18/24 · 6/12 | 23/24 · 11/12 | 16/24 · 4/12 |

## 两个配方与判定

SlowPath：Five-v1的学习率.005→.001，每阶段20→100，warm20，共520epochs；EarlySlowPath在此基础上warm20→5，共505epochs。C值保持.05/.2/.8/3.2/12.8，各100epochs。两项都是新组合，不是Legacy精确还原或证明最小的改动。

平方/初始化/dropout.043/仅中心化/非gate Adam WD.001、有效g坐标、canonical prox和最终|g| Top2不变。未改旧源文件、冻结版本或默认。

筛选要求四格坐标与完整折均不下降、每实例坐标不下降，且两个XOR维度坐标均严格提高；另单列两个XOR维度均满分的数值目标。不能用四格平均掩盖退化。

| 配方 | 值得新数据确认 | 两个XOR均24/24且保护Ring |
|---|---|---|
| SlowPath | False | False |
| EarlySlowPath | False | False |

## 每实例结果与辅助AUC

| 配方 | 任务 | p | 实例0/1命中（各/12） | Five实例0/1 | 最终测试AUC |
|---|---|---:|---|---|---:|
| SlowPath | xor | 256 | 1/2 | 0/2 | 0.529885 |
| SlowPath | xor | 512 | 0/0 | 1/0 | 0.496184 |
| SlowPath | ring | 256 | 7/10 | 7/11 | 0.720179 |
| SlowPath | ring | 512 | 12/11 | 9/9 | 0.965933 |
| EarlySlowPath | xor | 256 | 0/0 | 0/2 | 0.492756 |
| EarlySlowPath | xor | 512 | 0/0 | 1/0 | 0.485751 |
| EarlySlowPath | ring | 256 | 7/8 | 7/11 | 0.741338 |
| EarlySlowPath | ring | 512 | 9/7 | 9/9 | 0.795104 |

## 核验、成本与边界

- 正式96 selectors、49200epochs、688800 minibatch更新；训练函数累计6931.609秒（不含初始化、数据、I/O及审计），单条最长85.313秒。实际runtime逐条保存。
- 六条非证据随机GPU轨迹另计2290epochs；原Five入口逐位一致，两个新配方有/无设备观察器的模型/Adam/z/u/trace/RNG完全一致。
- 独立重建数据、列置换、六折、scaler、平方变换、初始化anchor、有效g分数与Top2；检查全部500epoch C路径、prox认证、Adam阶段步数、CUDA设备和全部最终预测。
- p256复用R62、p512复用R61的外层完整训练折。R63 LassoNet用内层90% fit；本表不能称配对超越LassoNet。未补跑已有Ring baseline。
- 本批使用已看过的数据做探索，六折有共享样本，不能声称独立确认、统计非劣或普遍优越。没有根据真值选epoch、切换任务配方或换seed重试。
- 不预设组合失败等于机制不存在，也不从组合成功推出某个单因素是原因。旧R58失败和DAG退化保持。
- 正式审计后总账3744+96=3840 selectors（3804 GRN线/控制+36 LassoNet），旧24次refit单列；本轮无refit。

## 产物

- [固定协议](../plans/PLAN_20260917_EMPIRICAL_COMBINATIONS_R67.md)
- [合同](../../results/ablation/empirical_combinations_r67_20260917/contract.json)
- [完整逐折结果](../../results/ablation/empirical_combinations_r67_20260917/run.jsonl)
- [汇总](../../results/ablation/empirical_combinations_r67_20260917/summary.json)
- [独立审计](../../results/ablation/empirical_combinations_r67_20260917/audit.json)
- [GPU重放](../../results/ablation/empirical_combinations_r67_20260917/preflight.json)
