# R03 Winner 来源分析

## 结论

R03 已完成并合并于 `971f4e0`。四个 R02 mixed winner 原样交给 Problem 2 官方评估器后，四组 Makespan 均低于对应 P2 baseline。当前证据支持这些 winner 至少包含可迁移到 P2 的方案质量，不是仅在 L2 条件下有效的 Cache 优化。

切图与分核的因素归因目前只覆盖 `case_014`、5 核、10 组：P3 partition 方案比 P2 partition 快 15.1379%；同一 partition 下，P2 与 P3 scheduler 生成的 plan hash 完全相同。因此这一个组合的收益由 partition 选择解释，未观察到 scheduler 参数的独立贡献。不能将此结论外推到另外三个 winner 或其他图。

## 四组 Winner 对照

改善率定义为 $r_2=1-T_2(x_C)/T_2(x_B)$；P3 mixed 相对 warm 的下降定义为 $r_3=1-T_3(x_C)/T_3(x_B)$。正值表示 Makespan 下降。

| 用例 | 核数 | Winner 来源 | P2 baseline $T_2(x_B)$ | P3 warm $T_3(x_B)$ | P3 mixed $T_3(x_C)$ | Winner 反测 P2 $T_2(x_C)$ | P2 改善 $r_2$ | P3 改善 $r_3$ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `case_014` | 2 | `current_fifo_cache_aware_list_g4` | 10,406,763 | 10,398,454 | 10,353,465 | 10,374,986 | 0.3053% | 0.4327% |
| `case_014` | 5 | `current_fifo_cache_aware_list_g10` | 6,751,802 | 6,715,169 | 5,698,636 | 5,746,752 | 14.8857% | 15.1379% |
| `case_034` | 4 | `current_fifo_cache_aware_list_g8` | 334,276 | 333,934 | 319,339 | 319,345 | 4.4667% | 4.3706% |
| `case_034` | 5 | `current_fifo_cache_aware_list_g10` | 326,605 | 326,166 | 304,036 | 304,273 | 6.8376% | 6.7849% |

四组反测均成功。P2 相对变化的原始数值、方案哈希及官方结果哈希见 [反向评估 JSON](results_round3/r03_winner_reverse_p2/r03-reverse-p2-20260923/reverse_validation.json) 和 [反向评估 CSV](results_round3/r03_winner_reverse_p2/r03-reverse-p2-20260923/reverse_validation.csv)。对应 R02 winner、P2 baseline 与 P3 warm 的必要来源方案和官方 JSON 收录在 [R02 来源压缩包](results_round3/r03_winner_reverse_p2/r03-reverse-p2-20260923/r02_source_bundle.tar.gz)；反测结果 JSON 收录在 [P2 官方结果压缩包](results_round3/r03_winner_reverse_p2/r03-reverse-p2-20260923/official_raw.tar.gz)。归档校验值记录在 [反测运行元数据](results_round3/r03_winner_reverse_p2/r03-reverse-p2-20260923/run_metadata.json)。

## `case_014` 五核 g10：Partition × Scheduler

下表中两种 scheduler 设置在各自的 partition 设置下生成完全相同的计划。

| Partition | Scheduler | plan hash | P3 Makespan | 新增搬运 bytes | 切分搬运 bytes | spill 搬运 bytes | Cache 命中率 |
|---|---|---|---:|---:|---:|---:|---:|
| P2 | P2 | `89f61e1684d1d8d0f2a94cf54dc37a94d4c67cf83816665a2353bdf35584b426` | 6,715,169 | 63,790,912 | 8,203,168 | 55,587,744 | 25.5131% |
| P2 | P3 | 与 P2/P2 相同 | 6,715,169 | 63,790,912 | 8,203,168 | 55,587,744 | 25.5131% |
| P3 | P2 | `a8bbefbadaf7e1f85a351732466663e9bf1057d4ffab72c2b7d5450d3a175393` | 5,698,636 | 68,526,400 | 16,686,496 | 51,839,904 | 21.8161% |
| P3 | P3 | 与 P3/P2 相同 | 5,698,636 | 68,526,400 | 16,686,496 | 51,839,904 | 21.8161% |

切换到 P3 partition 后，Makespan 减少 1,016,533 cycles（15.1379%）；新增搬运增加 4,735,488 bytes，切分搬运增加 8,483,328 bytes，spill 搬运减少 3,747,840 bytes，Cache 命中率下降 3.6969 个百分点。因此命中率或总搬运量都不能单独解释本次耗时变化。

按方案分配的运算周期代理量也显示 P3 partition 下负载更均衡：各核 PIPE_M 周期范围由 1,732,068–5,011,020 收窄为 3,292,956–3,565,620；PIPE_V 周期范围由 148,248–504,216 收窄为 297,528–356,040。这与 Makespan 下降相符，但这些是计划上的运算周期汇总，不是官方模拟器测得的忙碌时间、依赖等待、MTE 排队或 DDR 争用，不能据此确认具体时序机制。

四臂的官方 Makespan、搬运统计、方案哈希及运行指纹见 [2×2 结果 JSON](results_round3/attribution_case014_n5_g10/partition_schedule_ablation.json)、[子图与分配诊断](results_round3/attribution_case014_n5_g10/subgraph_diagnostics.json) 和 [运行元数据](results_round3/attribution_case014_n5_g10/run_metadata.json)。原始结果压缩包为 [官方 raw 归档](results_round3/attribution_case014_n5_g10/official_raw.tar.gz)。

## 解释边界

- 四个 winner 来自两个图、四个核数组合；它们在 P2 下全部改善，不代表 100-case 胜率，也不证明等评估预算下的算法优势。
- 只有 `case_014` 五核 g10 做了 2×2 归因。其他 winner 的 partition、mapping、ordering 与 Cache 时序贡献仍未拆分。
- 本次 2×2 中 scheduler 参数没有改变方案；这只说明固定 partition 与当前 `schedule_partition` 入口下两者等价，不说明 scheduler 对其他 partition 或图均无效。
- 官方 trace 曾生成并在本地保留，但未纳入压缩归档；本报告没有基于 trace 做忙碌区间或尾部等待分析。2×2 入口也未记录单独的方案生成墙钟时间。
- R02 的 warm 与 mixed 评估预算不同；本次反测回答的是固定 winner 在 P2 中的表现，不构成等预算对照。

## 下一步

优先复用现有 2×2 入口扩展 partition × scheduler 对照，先补 `case_034` 五核 g10 和 `case_019` 四／五核的代表性组合，再按预算扩到 `case_014`、`case_034`、`case_019` 的 4／5 核与 g8／g10／g12。固定图、核心数、组数、seed 和官方评估上限，逐臂记录方案哈希与 Makespan；对关键 warm/winner 选择性分析 trace，并补记构造耗时。

在矩阵显示可重复的 partition 贡献前，不先拆出多个 partitioner 模块，也不新增复杂结构聚合器。若 P2 可迁移收益在扩展组合中持续出现，再把有效 partition/schedule 构造抽成可组合候选；P3 Cache refinement 作为独立候选保留，不预设其单独有效。