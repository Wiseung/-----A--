# Round 2 仓库审查

## 审查范围

- 当前代码快照仍为 `main` 上的 `30a62dd28cb69d475ca7862fe405aa1e6a493711`（`first commit`）。审查对象是实际工作区，不是泛化建议。
- 工作区中原有的 `CODEX_PHASE2_REPO_SPECIFIC.md` 保持未修改。
- 官方目录未改动。根目录多核附件 ZIP 共 114 个文件；逐文件 SHA-256 对照后，ZIP 与 `2026_official/` 中 114 个正式 README、代码、数据、配置和文档文件一致。解压目录另有 9 个不在 ZIP 内的 `__pycache__/*.pyc` 文件，本轮未删除。
- 工作区选中的 `.venv` 是 Python 3.9.6，不能运行仓库现有的 `dataclass(slots=True)` 代码。验证使用本机 Python 3.11；README 已同步说明。

## 原审查中已复现的问题

- [solver/schedule_common.py](solver/schedule_common.py)：问题 2/3 的候选评分完成时间未保存，提交给后继的时间只取 Pipe 结束时刻。
- 问题 2/3 的代理此前未把源端 MTE3 写出、目标端 MTE2 读入和实际输入可用时刻完整串起来；通信次排序键对候选核心不敏感。
- Cache 代理以前在构造顺序上立即插入未来 miss，并可能提前淘汰旧条目。
- [solver/improve.py](solver/improve.py)：迁移一个子图时数值排序会重排核心上其余子图；预算小的时候 merge 邻域可能得不到评估机会。
- [solver/solver.py](solver/solver.py)：基准时间重复扣除、候选超时可超过 CLI 单次评估上限、失败候选可能覆盖 `best`、汇总按 case/problem/core 覆盖不同运行。
- [experiments/collect_results.py](experiments/collect_results.py)：缺失统计只以 summary 中出现的 case 为分母。

## 本轮落地

- 评分和提交共享 `PlacementEstimate.group_finish`；问题 2/3 用局部操作时序与候选核心专属通信增量，记录读/写 COPY、跨核 COPY、重复输入和每核 M/V 工作量。
- 场景 B 微图按每个实际 source-core→target-core 连接计一对 COPY；独立计入最终输出写回。Cache 查询只回放查询时刻以前完成的插入事件。
- 增加问题 2→3、旧方案、少核方案追加空核、单组保底和关键路径/释放字节拓扑序候选；正式方案只由成功官方评估更新。
- 增加 run/candidate ID、图/配置/官方/solver 指纹、预算字段、候选账本、逐子图诊断和按 manifest 的完整汇总。
- 局部邻域迁核保序；move、merge、swap、reinsert、split 轮转生成。

## 尚未完成

- 尚未运行 100-case 正式矩阵，也未为 `case_014` 单独延长预算分析单核参考。
- 尚未实现非连续分支聚合/区域增长、基于官方 `cache_events` 的 Cache 顺序后处理、生存期驱动排序，以及经口径核对的 `physical_ddr_bytes_derived` 外部统计。
- 新的传输时序仍是确定性代理，不模拟官方全局 DDR 公平带宽竞争；官方 Makespan 仍是唯一性能结果。
