# Round 2 实验结果

## R00：固定方案 2×2 对照

每个 case 使用旧的 Problem 2 方案 `x_B` 和旧的 Problem 3 方案 `x_C`，分别调用无 L2 与 L2 官方评估器。表中 makespan 单位为 cycles；指标定义为 `T_B(x_B)/T_C(x_B)`、`T_C(x_B)/T_C(x_C)` 和 `T_B(x_B)/T_C(x_C)`。

| Case / 核数 | `T_B(x_B)` | `T_C(x_B)` | `T_B(x_C)` | `T_C(x_C)` | 固定方案硬件效果 | Cache 下算法适配 | 综合效果 |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_019 / 4 | 42,158 | 42,158 | 78,089 | 78,089 | 1.000000 | 0.539871 | 0.539871 |
| case_034 / 4 | 420,981 | 420,981 | 525,396 | 525,396 | 1.000000 | 0.801264 | 0.801264 |
| case_014 / 4 | 8,636,778 | 8,631,067 | 11,156,087 | 11,145,703 | 1.000662 | 0.774385 | 0.774898 |

- case_019：固定 Problem 2 方案的 Cache 命中率约 2.28%，makespan 不变；旧 Problem 3 方案在两个评估器下都为 78,089 cycles，Cache 命中率为 0%。
- case_034：旧 Problem 3 方案 Cache 命中率约 9.23%，makespan 仍为 525,396 cycles；Problem 2 方案在两种评估器下均为 420,981 cycles。
- case_014：固定 Problem 2 方案在 L2 下只减少约 0.066% makespan；旧 Problem 3 方案仍显著慢于 Problem 2 方案。分区与 spill 流量记录在固定方案对照 JSON 中。

机器记录：
- [三例固定方案对照](results_round2/r00_fixed_plan_cache/c4b451810e36/fixed_plan_cache_comparison.json)

PR 证据包包含三例对照摘要和归档方案；case_019 的四份官方原始结果、R01/R02 官方原始结果会一并提交。case_014/case_034 的 R00 原始 JSON 与大型 Trace（约 63 MB）留在本地 `results_round2`，避免把完整 Trace 包推入代码 PR；对照 JSON 保留 Makespan、搬运分项、Cache 指标、plan hash 和官方代码 hash。

## R01：切分与分核策略分离

case_019、4 核、8 子图，四种方案都由 Problem 3 官方评估器评估。

| Partition | Placement proxy | Makespan | 活跃核心数 | 新增 COPY bytes | Cache hit rate |
|---|---|---:|---:|---:|---:|
| p2 | p2 | 26,947 | 4 | 68,136 | 10.89% |
| p2 | p3 | 26,947 | 4 | 68,136 | 10.89% |
| p3 | p2 | 56,214 | 2 | 65,060 | 0% |
| p3 | p3 | 56,214 | 2 | 65,060 | 0% |

同一切分下，p2/p3 分核代理产生相同 plan hash；本实验的主要差异来自切分结构。p3 切分只激活两个核心，p2 切分激活四个核心。逐核 M/V 周期与逐组候选评分见诊断文件。

- [R01 汇总](results_round2/r01_partition_schedule/7b5c8f3a9961/partition_schedule_ablation.json)
- [R01 子图诊断](results_round2/r01_partition_schedule/7b5c8f3a9961/subgraph_diagnostics.json)

## R02：候选保底与拓扑序候选

当前代码在 case_019、4 核上先评估历史方案，再评估新候选；Problem 3 先评估 Problem 2 warm-start。

| 评估问题 | 候选家族 | 官方 Makespan | 新增 COPY bytes | 结果 |
|---|---|---:|---:|---|
| p2 | legacy history | 42,158 | 41,508 | 保留为候选，不是最终最好方案 |
| p2 | 当前 ID 拓扑序 | 26,947 | 68,136 | 接受 |
| p2 | critical-path 拓扑序 | 58,645 | 855,040 | 拒绝 |
| p2 | release-bytes 拓扑序 | 46,890 | 215,268 | 拒绝 |
| p3 | p2 warm-start | 26,947 | 68,136 | 接受 |
| p3 | 当前 ID 拓扑序 | 56,214 | 65,060 | 拒绝 |
| p3 | critical-path 拓扑序 | 58,355 | 855,040 | 拒绝 |
| p3 | release-bytes 拓扑序 | 66,398 | 245,860 | 拒绝 |

在这个 case 上，拓扑排序变体没有优于 ID 顺序；更重要的是 Problem 2 方案经 Problem 3 官方评估后保住了 26,947-cycle 结果，避免 Cache 原生候选退化。这个结果只代表 case_019 与本次预算，不是跨用例结论。

- [R02 汇总](results_round2/r02_topology_case019_final/457738c42ffb/summary.json)
- [R02 候选账本](results_round2/r02_topology_case019_final/457738c42ffb/candidate_trials.jsonl)
- [R02 子图诊断](results_round2/r02_topology_case019_final/457738c42ffb/subgraph_diagnostics.jsonl)
- [R02 完整用例清单](results_round2/r02_topology_case019_final/457738c42ffb/complete_case_manifest.json)

## 运行边界

本轮未运行完整 100-case × 3 问 × 2–5 核矩阵，也未完成 case_014 官方单核入口的延长预算剖析。固定方案对照覆盖三个代表用例，R01/R02 的搜索实验仅覆盖 case_019。没有预设或宣称固定百分比提升；所有性能值均来自记录的官方评估器输出。
