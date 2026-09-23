# Round 2 修复报告

## 代码变更

- [solver/schedule_common.py](solver/schedule_common.py)：候选结果显式保存评分时的 group start/finish；问题 2/3 加入局部依赖排布、MTE3 写出、MTE2 读入、固定跨核等待和候选核差分 COPY 估计；Cache miss 按预测完成时刻重放。可选诊断包含逐子图逐核心分数、COPY 字节、Pipe 结束时刻、输出可用时刻和重算后的 partition COPY 对照值。
- [solver/partition.py](solver/partition.py)：rank 沿实际子图 DAG 逆拓扑计算；加入 ID、关键路径、分支局部性和释放字节拓扑序。
- [solver/improve.py](solver/improve.py)：迁核保留未迁移子图的相对顺序；邻域家族轮转，提供 move、swap、reinsert、split 和 Problem 1 merge。
- [solver/solver.py](solver/solver.py)：分开记录基准/搜索/候选评估/总墙钟，限制单候选超时；`legal_draft` 与 `best_validated` 分开；加入 Problem 2→3 warm-start、少核扩展和单组保底，候选来源与诊断分别写入 JSONL。
- [solver/run_identity.py](solver/run_identity.py)：提供 run ID、输入/代码指纹与 run metadata 校验。
- [experiments/run_all.py](experiments/run_all.py)：默认写入独立 `results_round2/<experiment-id>/<run-id>/`；resume 要求同一 run ID、预算、代码/配置指纹与矩阵 manifest。
- [experiments/collect_results.py](experiments/collect_results.py)：按 batch/case manifest 统计完整计划矩阵；区分 success、timeout、invalid、evaluation failure、not-run 和单核参考缺失；不回写原始 summary。
- [experiments/make_figures.py](experiments/make_figures.py)：Problem 3 两条曲线与 Cache ratio 都只按同 run、同 case、同配置的配对集计算。
- 新增固定方案 2×2 对照和分区/分核消融脚本；README 已更新到新路径和 resume 用法。

## 回归验证

- 命令：`python3.11 -m unittest discover -s tests -v`
- 结果：33 项通过；其中原有 13 项通过，其余覆盖完成时刻一致性、候选核心通信差分、官方 COPY 计数、输入可用时刻、未来 Cache 插入、候选保底、best 不覆盖、run 隔离、局部邻域、拓扑序候选和 100-case manifest 分母等。
- 对所有修改的 Python 文件运行 Pylance 诊断：未发现错误。
- 额外端到端运行：case_019 / 4 核，Problem 2 和 Problem 3，产物见 `results_round2/r02_topology_case019/5d807967d64f/`。

## 兼容性与边界

- 当前虚拟环境 `.venv` 是 Python 3.9.6，会在导入现有 `dataclass(slots=True)` 代码时失败；本轮使用 Python 3.11 验证，没有修改虚拟环境。
- 未修改 `2026_official/code`、`2026_official/data` 或旧 `results/`。
- 未运行完整 100-case 矩阵；本轮性能结论仅来自下述固定方案和代表性 case 实验。
