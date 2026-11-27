# Deep-Regularization 仓库约定



Research evidence and reproducibility notes. Consult the retained research index and existing execution restrictions below.



## 研发入口



- [项目首页](README.md) 维护用途、安装和使用方式。

- [研发文档导航](docs/README.md) 定位现行设计、计划与历史记录。



## XOR / Ring thread integrated — 2026-09-20



The user requested summarizing this research thread into main. The bounded integration contains documentation and evidence snapshots, not a replacement of main's model implementation. Start with [the thread summary](docs/research/20260920-xor-ring-thread/README.md), [meeting report](docs/research/20260920-xor-ring-thread/MEETING_REPORT.md), and [provenance](docs/research/20260920-xor-ring-thread/PROVENANCE.md).



- New experiments have been paused by the user since 2026-09-18. This documentation integration does not authorize training, inference, tuning, retries, paper rewrites, or default promotion.

- The audited research ledger is 4080 selectors; 24 earlier selected-support refits are separate.

- GRN-FS-Five-v1 and Five-SlowPath-v1 are retained research candidates. Ring gains are established in their stated comparisons, but the original joint non-regression gate and minimality claim remain unmet. High-dimensional XOR and DAG limitations must remain visible.

- Legacy-poly2 (old Ring full polynomial configuration) and WN-legacy (raw-input historical path) are different references. Do not splice their results into one curve or claim exact historical runtime recovery.

- Gate/score theory, A/D controls and passive alternative-score diagnostics were already discussed on 09/06. An active gradient-score feedback algorithm remains unimplemented; distinguish it from completed diagnostics.

- Use Top-k recovery as the primary metric. Preserve dataset identities, pairing limitations, failed screens, and the distinction between retained versions and promoted defaults.

- Source snapshots preserve original bytes and may contain links specific to their original workspace. They are evidence, not runnable main modules. Full checkpoints and runtime artifacts remain in the integration research workspace.



The original research workspace keeps retained research under `research/legacy-migration/`. Do not alter or retire that workspace as incidental cleanup. This main worktree contains a portable report/evidence package with a standard-library verifier; see its README for the verification command.



## Workspace 采用记录



- 来源 [Workspace 098bd2e](https://github.com/DreamSoul-AI/workspace/tree/098bd2eccf23eb1e6d77a3eec6db51e6bc387daf)。采用范围为Learnings 命名与分发、文档规范配置、长作业检查点与恢复、调研、研究计划与研究结论归档。项目专用流程见本文件。

- 单项更新：[Workspace 65657c5](https://github.com/DreamSoul-AI/workspace/tree/65657c50cf6d98ac1898ae0183cb9ca98498e0a9)。采用旧对话重读、新对话定位、文档主动补链接、浏览器收尾、飞书专节及 CLI 任务临时目录、共享资源主动检查；保留项目专有约定和原整版基线。共享 data/models 直接复用，仅 artifacts 按仓库隔离，接入示例见 [Workspace 教程](https://github.com/DreamSoul-AI/workspace/blob/main/templates/software-engineering/examples/shared-resources/README.md)。

- 单项更新：[Workspace f01fab8](https://github.com/DreamSoul-AI/workspace/tree/f01fab8598f570a45eca8320e546d636268b700a)。采用通用文档规范及飞书/GitHub 平台章节入口、首次接入默认值和已有选择保留、代码架构设计与代码结构设计并列命名；图表简洁要求随通用规范读取。保留本项目专有约定、既有文档路径及原整版基线。[规范分发副本](https://github.com/DreamSoul-AI/workspace/blob/f01fab8598f570a45eca8320e546d636268b700a/%E6%96%87%E6%A1%A3/%E8%A7%84%E8%8C%83/%E9%80%9A%E7%94%A8/%E9%80%9A%E7%94%A8%E6%96%87%E6%A1%A3%E8%A7%84%E8%8C%83.md)。

- 单项更新：[Workspace b21fec8](https://github.com/DreamSoul-AI/workspace/tree/b21fec8aad842db7b8fd11042554e560dbe4b49d)。共享资源仓内入口推荐 `.shared/data`、`.shared/models`、`.shared/artifacts`；数据和模型按资源组织，产物按项目隔离，路径仍按项目配置。保留原整版基线及项目约定。
- 单项更新：[Workspace 823d496](https://github.com/DreamSoul-AI/workspace/tree/823d4965035899d66a2d57751f1d28fbf934f99a)。采用临时产物与共享入口分工、资料根定位、目录升级检查、双端文档先改本地及面向读者的汇报；源码、测试、项目 docs 与专有约定保留，原整版基线继续沿用。
- 单项更新：[Workspace ea16de5](https://github.com/DreamSoul-AI/workspace/tree/ea16de5eda676aadaeefc1897ce0b83c062de2b6)。采用常规问题直接处理、结合目标和进度主动建议下一步；Knowledge 能力更新沿 [Workspace 教程](https://github.com/DreamSoul-AI/workspace/blob/main/资料/教程和工具/使用教程.md#workspace-与-knowledge-更新) 默认接入，保留通用与个人来源、项目专有约定和原整版基线。
- 单项更新：[Workspace f020aa5](https://github.com/DreamSoul-AI/workspace/tree/f020aa5b6965766f7308fa31b0836e478c9da914)。采用加粗只包文字、所有标点放在外面的统一写法，移除未采用的本机配置默认目录及专门忽略项；配置位置和相应保护沿用项目约定，保留其他规则及原整版基线。
- 单项更新：[Workspace a9da11e](https://github.com/DreamSoul-AI/workspace/tree/a9da11e30412aa3ec52a88cf76150da129b5063b)。文档按用途和目标读者组织正文，补齐独立阅读所需的背景与依据；保留原整版采用基线及项目约定。
- 本轮适用更新：[Workspace 84f9696](https://github.com/DreamSoul-AI/workspace/tree/84f969679eb06b7d742ee7b384c13e371df9d425)。核对并合并当前适用的共用规则，补充新对话项目选择和按场景理解输入的约定，保留项目专有内容、个人配置、现有布局及原整版采用基线。

## 候选一代码与排名初测，2026年11月27日

本次用户明确授权建立独立分支、修复代码并测试候选一，只做 feature ranking。新增[运行入口](experiments/ranking_taylor_fisher.py)及[实现与结果记录](<docs/研究记录 Research Record/2026年11月27日 候选一 Ranking 初测.md>)。已完成128维 XOR/Ring 的配对初测；这项授权不扩展为候选二至四、下游重训、完整参数搜索或默认版本提升，原历史证据不改写。
