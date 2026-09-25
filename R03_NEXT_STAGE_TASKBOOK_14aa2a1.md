# R02-full 仓库核对与第三阶段增量任务书

审查基准：`main` 提交 `14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c`
提交说明：`feat: stabilize Problem 3 warm-start and provenance`
目标仓库：`Wiseung/-----A--`
本文件是开发与实验任务书，不是已完成的代码修改或新增实验报告。

## 0. 审查范围与证据边界

已读取该提交中的 `solver/solver.py`、`solver/evaluate_adapter.py`、
`solver/partition.py`、`solver/cache_aware.py`、
`tests/test_solver_reliability.py`、
`experiments/problem3_warm_start_ablation.py`、
`experiments/partition_schedule_ablation.py`，以及 R02-full 的 CSV、JSON、运行元数据和目录树。

下列周期数来自仓库提交的实验汇总，不是本次独立重跑的官方评估结果。
本次没有复跑单元测试，也没有重新执行官方评估器。

当前提交的
`results_round2/r02_problem3_warm_start/r02-full-20260923/`
仅含 `batch_manifest.json`、`run_metadata.json`、
`problem3_warm_start_ablation.csv`、
`problem3_warm_start_ablation.json` 四个文件。
JSON 中引用的本轮 `schedules/`、官方 `raw/` 和候选账本没有随该目录提交。
这不否定本地实验，但当前远程归档还不足以逐方案复验。

## 1. 已核对的 R02 数值

单位：cycles。“下降”定义为 `1 - mixed / warm`，正值表示混合更快。

| 用例 | 核数 | 历史 P3 重评 | 本轮 P2 | P2→P3 warm | 混合 P3 | 混合相对 warm 的耗时下降 |
|---|---:|---:|---:|---:|---:|---:|
| case_014 | 2 | 13,321,252 | 10,406,763 | 10,398,454 | 10,353,465 | 0.4327% |
| case_014 | 4 | 11,145,703 | 6,395,433 | 6,372,134 | 6,372,134 | 0.0000% |
| case_014 | 5 | 7,099,894 | 6,751,802 | 6,715,169 | 5,698,636 | 15.1379% |
| case_019 | 2 | 79,137 | 41,417 | 41,417 | 41,417 | 0.0000% |
| case_019 | 4 | 78,089 | 26,947 | 26,947 | 26,947 | 0.0000% |
| case_019 | 5 | 71,649 | 26,588 | 26,588 | 26,588 | 0.0000% |
| case_034 | 2 | 638,004 | 419,344 | 419,344 | 419,344 | 0.0000% |
| case_034 | 4 | 525,396 | 334,276 | 333,934 | 319,339 | 4.3706% |
| case_034 | 5 | 402,601 | 326,605 | 326,166 | 304,036 | 6.7849% |

四个严格改善的最佳来源全部是 `current_fifo_cache_aware_list_g*`：
- case_014 / 2 核：g4；
- case_014 / 5 核：g10；
- case_034 / 4 核：g8；
- case_034 / 5 核：g10。

本轮元数据为：
`p2_max_evals=2`、`warm_max_evals=1`、
`mixed_max_evals=2`、`improve_enabled=false`、
`time_limit_sec=300`、`baseline_timeout_sec=2`、
`evaluator_timeout_sec=600`、`retain_traces=false`、`seed=0`。

因此，本轮验证的是“P2 已验证方案与当前原生 P3 构造候选的互补性”；
不能写成“Cache 局部后处理已经取得收益”，也不能写成相同评估预算下优于旧求解器。
`p3_original` 实际是历史方案在当前 P3 评估器下重评，不是旧版求解器等预算重跑。

## 2. 需要据此更新的策略

保留 P2 已验证方案作为 P3 的重要起点，同时保留当前原生 P3 候选。
不要把 P3 强制收缩成 cache-only refinement。

case_014 / 5 核：
warm 为 6,715,169，mixed 为 5,698,636，下降 15.1379%；
Cache 命中率从 25.5131% 降至 21.8161%；
额外搬运量从 63,790,912 增至 68,526,400 bytes。
这一观察说明命中率、额外搬运量不能独立代替 Makespan，
但尚未确定收益究竟来自切图、分核、顺序、spill 还是共享带宽时序。

case_019 / 4 核的当前 warm 基准是 26,947，不应继续用旧 smoke 的 42,158
作为下一轮性能比较基线。不同 run 必须带方案哈希和运行身份。

## 3. 总约束

只修改自有 solver、实验脚本、测试和报告。
不得修改官方评估器和正式硬件配置；不得改变原图操作周期和节点身份。
继续复用现有合法性检查、指纹、官方 subprocess 评估、候选账本及分项计时。

不要再从头开发 warm-start、来源记录或已有五类邻域。
不预设某个新算法必然提升固定百分比。
新候选可以失败或退步，但不可覆盖更好的正式保底结果。

## 4. 第一批：补齐轻量复验材料

从本地原有产物中补齐，而不是重跑完整 R02：

1. 本轮每个最佳方案的不可变 `candidate_id.json`。
2. 该方案对应的官方结果 JSON，及图、配置、官方代码、方案哈希。
3. `candidate_trials.jsonl`；保留不被接受的候选记录。
4. 记录每个试验臂引用哪个不可变方案，不依赖后续可能更新的
   `*_best_validated.json` 作为唯一来源。
5. 验证 CSV/明细中的 `plan_path`、`source_plan`、`candidate_id` 能被解析到真实归档。
6. 大型 trace 无需全部提交。若官方 raw JSON 本身很大，允许压缩归档；
   轻量摘要必须说明来源及校验哈希，不能把摘要标为完整原始输出。

验收：从新目录取得仓库与这些轻量产物后，能够复验四个胜出方案；
不依赖开发者机器上的绝对路径。

## 5. 第二批：修补预算与保底的两个边界

### 5.1 将立即构造候选改为按需构造

当前 `solve_one()` 先执行多个 `generate(...)` 并组装 `candidate_specs`，
之后才进入正式评估循环。即使 `max_evals=1` 且 P2 已验证方案存在，
仍可能先构造多个不会被评估的候选。

改成候选描述或工厂函数的惰性序列：

加载已验证起点 → 先评估并保存起点 → 检查剩余预算 →
构造下一候选 → 验证及评估。

构造前后都检查 deadline；不要把全部构造开销挤到第一条保底评估之前。
优先保留现有 `try_candidate` 的合法性、隔离和计时逻辑，
避免为了惰性生成重写评估器。

新增测试：
- 已有可信 P2 方案且 P3 `max_evals=1` 时，其他候选构造器不被调用。
- 后续构造器超时、失败或预算耗尽，不会使已完成的 warm 结果丢失。
- 构造耗时进入求解开销，不混入官方模拟 cycles。

### 5.2 区分“候选已加入”和“保底已验证”

当前少核方案追加空核的候选位于原生候选及若干拓扑候选之后。
两次评估预算可能根本到不了这类候选。

本轮已有明确诊断例：
case_014 的 P2 四核为 6,395,433，五核为 6,751,802；
五核构造结果慢约 5.5722%。
这不能证明第五核本身有害，应检查四核方案追加空核后在五核下的真实成绩，
以及该候选是否获得评估机会。

保留两种实验模式，不要混淆：
- 原 R02 两候选消融：严格 warm + native，便于复验旧实验。
- 正式求解模式：为兼容的同场景历史最优、P2→P3 起点、
  少核扩展候选设置明确处理规则。预算容不下全部候选时，
  记录 `mandatory_candidate_deferred`，不得声称全部保底已生效。

不能凭空保证两次评估同时覆盖三类不同方案。
允许提高正式模式的最低评估配额，或复用完全匹配的同场景已验证评分，
但跨场景及核数变化必须按既定官方复验规则处理。

新增测试应使用有多个不同候选的图；仅用链图可能因候选去重掩盖顺序问题。

### 5.3 同 run 重试的附加回归

`best_result` 在每次 `solve_one` 调用时重置，
已接受候选写入同一个 `run_id` 的 `best_validated` 路径。
对 P3 增加回归：同 run 已有更好的原生 P3 结果时，
再次用小预算执行较差的 P2 warm，不能使用于最终发布的 incumbent 退步。

这是静态路径风险，不是声称本轮 R02 已经发生数据污染。
各消融臂可有各自的 run-best，但共享正式 incumbent 应单独保护。
现有“评估失败不覆盖”测试不等价于“评估成功但更差也不覆盖”。

## 6. 第三批：先解释现有四个胜出方案

### 6.1 胜出方案反向验证

将四个混合胜出方案原封不动交给 Problem 2 官方评估器。
不生成新方案，不做局部搜索，不依赖单核加速比。

对每个组合填表：

| 方案 | P2 无 L2 | P3 有 L2 |
|---|---|---|
| 本轮 P2 最佳方案 x_B | 已有 T2(x_B) | 已有 T3(x_B) |
| 本轮 mixed 胜出方案 x_C | 新增 T2(x_C) | 已有 T3(x_C) |

新增目标最多四次唯一正式评估；选择性 trace 重放另行记账。

解释规则：
- 若 x_C 在 P2 下也更快，说明至少存在可迁移的通用方案质量改善，
  值得将此构造方式作为 P2 的候选之一。
- 若只在 P3 下有收益，则继续分析 Cache/DDR 与调度时序的交互。
- 以上都不能单靠命中率直接判断。

### 6.2 复用现有切图 × 分核 2×2 脚本

先做 case_014 / 5 核 / 10 组，再做 case_034 / 5 核 / 10 组。
保持节点 ID、组数和 P3 官方评估场景一致。

比较：
P2 切图 + P2 分核；
P2 切图 + P3 分核；
P3 切图 + P2 分核；
P3 切图 + P3 分核。

当前 `partition.py` 中 P2/P3 的 cut_weight 分别为 0.12 / 0.05，
`cache_aware.generate_schedule()` 同时改变切图和分核的 problem 参数。
所以原生 P3 胜出不能直接归因于 Cache 分核代理。

现有入口可直接运行：
```bash
python3.11 experiments/partition_schedule_ablation.py \
  --case case_014 --ncores 5 --group-count 10 \
  --run-id r03-attribution-case014-n5-g10 \
  --results-dir results_round3/attribution_case014_n5_g10

python3.11 experiments/partition_schedule_ablation.py \
  --case case_034 --ncores 5 --group-count 10 \
  --run-id r03-attribution-case034-n5-g10 \
  --results-dir results_round3/attribution_case034_n5_g10
```

该脚本当前 experiment_id 固定为 `r01_partition_schedule`；
如增加新的外层阶段名称，不要误改历史 run 身份。
该入口当前仍按 Evaluator 默认保留 trace；运行大图前决定是否新增显式开关。
这些命令是待执行实验，不是本次已执行内容。

新增记录：实际组数、组工作量分布、各核 M/V 负载、
划分搬运、spill 搬运、官方 Makespan、Cache 字节统计、
生成耗时、评估耗时、方案哈希。
必要时仅对 warm 与 winner 重放 trace，提取各核各 Pipe 的忙碌区间和尾部等待。

## 7. 第四批：case_014 粒度与切点实验

先保持分核器不变，不同时加入新 Cache 后处理和新负载惩罚。

第一阶段：4/5 核，group_count = 8, 10, 12, 16。
必须包含 g10，因为本轮已有 g10 的胜出参照。
第二阶段根据第一阶段曲线选择性补 4、24、32，
不要一次对所有参数做笛卡尔积搜索。

同时保留“只改组数”和“只改切点规则”两种可归因实验。
当前切点平衡项除以全图工作量；新的局部归一化、工作量窗口应作为独立开关，
不能与新分核评分同时改动。

P2/P3 中子图边界不等于 Task 边界；
同核增加子图数本身不会自动清空缓存。
切分收益必须从最终分核、官方顺序、live range 或 spill 变化中解释。

不要把各核最后完成时刻相近当作独立优化目标：
需区分计算负载不均、上游依赖等待、MTE 排队和 DDR 争用。

## 8. 第五批：结构化候选，不重做已有拓扑策略

当前已有 `id`、`critical_path`、`branch_locality`、`release_bytes`
四种拓扑入口；主候选流程只显式加入其中两个替代拓扑。
先让已有策略获得可控评估机会，再新增结构聚合器。

需要明确：
- 改一个拓扑优先级，不等于已经实现非连续分支聚合。
- 当前 `branch_locality` 是静态优先级，不是完整独立分支保持器。
- 当前 `release_bytes` 根据原拓扑序估计最后消费者；
  如要宣称动态释放感知，需要按实际剩余消费者状态更新，并单独消融。

新结构算法优先做一个可核验版本：
保持可并行分支分离，沿高通信链进行有界聚合；
共享输入以张量—消费者集合表示，不展开消费者完全图。
每次聚合检查收缩后的子图 DAG，不能只检查原图无环。
分核与核内顺序也需通过现有整体合法性检查。

比较时固定实际组数、分核器、官方调用上限、种子和后处理开关。
同时公布每个构造器的单独结果与加入 incumbent 后的组合结果，
避免最优保底把劣质候选的真实表现全部隐藏。

## 9. 实验口径与运行环境

严格区分以下量：
- T2(x_B)：无 L2 下本轮 P2 方案；
- T3(x_B)：同一方案在有 L2 下重评；
- T3(x_final)：混合 P3 最终方案。

只要 x_B 已完成 P3 官方评估并保留在候选集合中，
该次运行选择最小 Makespan 可保证 T3(x_final) <= T3(x_B)。
不能仅凭 warm-start 机制保证 T3(x_final) <= T2(x_B)；
本轮跨场景 9/9 不退步属于实测结果。

单核参考缺失不影响固定核数的多核方案比较。
将 case_014/case_034 的官方单核参考补算作为独立任务，
设置更长的受控预算，保存阶段耗时，保持官方入口不变。
2 秒超时不能解释为图不可解，也不能用任意优化单核结果补分母。

`evaluator_calls` 当前包含该次 solve 内的单核基准调用；
表中的 3 次可能是单核 1 次 + 多核候选 2 次。
新增 `baseline_eval_calls`、`candidate_eval_calls`，
并披露 warm-start 的上游 P2 生成成本：
有现成 P2 的增量成本，与从零完成 P2+P3 的成本分别报告。

`retain_traces=false` 只控制持久化复制；
当前适配器仍向官方程序传入 `--trace-output`。
不要将“不归档”写成“没有生成/序列化 trace”。
`Evaluator.__init__` 默认仍为 true，R02 脚本显式传 false；
若要全工程默认不留 trace，应让所有入口显式采用一致策略。

## 10. 覆盖与交付

本轮只有三个图、九个核数组合，不能描述为九个独立测试图。
在完成上述小批实验后，将回归扩到事先固定的 8–12 个不同结构用例，
随后开始低预算 100-case 普查；完整三问、全部核数的高预算矩阵放在后续。
不要长期以“算法还未最终稳定”为理由完全不扩展覆盖。

交付：
- 独立的小提交：轻量归档、惰性候选/保底边界、归因实验、粒度实验。
- 新增回归测试与真实执行日志。
- `ROUND3_ATTRIBUTION.md`：说明四个 winner 在 P2/P3 下的差别。
- 可解析的 manifest、候选账本、方案、官方结果与分项耗时。
- 每个实验明确“已执行/失败/未执行”，记录负结果，不虚构预期收益。

下一阶段完成标准不是增加三个算法文件名，
而是解释现有四个 winner，并在不破坏保底的前提下增加真正不同的有效候选。

## 来源（固定到审查提交）

- [R02 CSV](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/results_round2/r02_problem3_warm_start/r02-full-20260923/problem3_warm_start_ablation.csv)
- [R02 JSON](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/results_round2/r02_problem3_warm_start/r02-full-20260923/problem3_warm_start_ablation.json)
- [R02 运行元数据](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/results_round2/r02_problem3_warm_start/r02-full-20260923/run_metadata.json)
- [求解主流程](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/solver/solver.py)
- [评估适配器](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/solver/evaluate_adapter.py)
- [划分器](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/solver/partition.py)
- [P3 构造入口](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/solver/cache_aware.py)
- [四臂脚本](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/experiments/problem3_warm_start_ablation.py)
- [切图/分核消融脚本](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/experiments/partition_schedule_ablation.py)
- [可靠性测试](https://github.com/Wiseung/-----A--/blob/14aa2a1fdb0bd7eb9e144dd5843ff369ef01d28c/tests/test_solver_reliability.py)
