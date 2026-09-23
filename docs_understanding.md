# 2026 官方评估器执行语义核对

核对依据：已解压的官方附件 `2026_official/`。结论以 Python 源码执行路径为准，文档用于辅助理解。

## 输入、配置与方案校验

- 原图 JSON 顶层含 `ops`、`tensors`、`edges`。所有 op/tensor ID 在同一整数空间内全局唯一且非负；边端点有效、边不重复、完整原图无环。四条合法 Pipe 为 `PIPE_MTE2`、`PIPE_MTE3`、`PIPE_M`、`PIPE_V`。[校验入口](2026_official/code/evaluation_validation.py#L171)
- 源码允许直接 `op -> op` 边（调度器将其按 `data_size` 处理），尽管题面主要描述 Tensor/Op 二部图。它禁止 `tensor -> tensor`，而不是禁止所有直接 Op 边。[原图校验](2026_official/code/evaluation_validation.py#L217) [问题2直接边处理](2026_official/code/multicore_cut_evaluate_problem_2.py#L218)
- 调度方案顶层恰好是 `node_to_subgraph`、`core_schedules`。映射必须恰好覆盖非 `COPY_IN/COPY_OUT` 操作；key 可为整数或 ASCII 十进制字符串，转换成整数后仍须唯一；sgid 为非负整数。每个 sgid 在非空的核心列表中恰好出现一次，核心子列表允许为空。源码同时检查子图 DAG 和同核依赖顺序。[方案校验](2026_official/code/stub_multicore_cut_and_schedule.py#L114)
- 方案入口检查原图 DAG；问题1再将子图依赖与同核 Task 顺序合并检查，问题2/3检查本地数据/内存依赖、Pipe FIFO 和跨核 COPY 的全局操作图是否有环。[Task 顺序校验](2026_official/code/evaluation_validation.py#L252) [执行图校验](2026_official/code/evaluation_validation.py#L261)
- 配置必须从 `data/config.txt` 或显式 `--config` 读取，无代码默认配置；当前固定值为 L1=524288、UB=131072、DDR=60 B/cycle、问题1等待100/1000 cycles、问题2/3跨核同步500 cycles、问题3 L2=1048576 bytes、250 B/cycle。[固定配置](2026_official/data/config.txt#L1) [配置解析](2026_official/code/evaluation_validation.py#L26)
- 单核 Pipe 槽位由代码固定为每条 Pipe 同时1个在途 op。普通 op 实际 duration 为 `max(1, cycles)`；COPY 的 duration 为其端点 tensor 大小总和除以带宽后向上取整且至少1 cycle。DDR 参与条件是 COPY 任一端点 tensor 位于 DDR。[Pipe 与 duration](2026_official/code/schedule_step3.py#L29) [duration/DDR 判定](2026_official/code/schedule_step3.py#L74)

## Problem 1：场景 A

- 每个子图独立成为一个 Task，核心与 Task 顺序来自 `core_schedules`。Task 边界数据经 DDR 中转；同核不同 Task 也不保留 L1/UB 内容。原始 DDR 输入/最终输出处理与中间 tensor 边界处理按 tensor 及 Task 的 producer/consumer 集合构造 COPY，不按每条二部图边重复计费。[Task 构造](2026_official/code/multicore_cut_evaluate_problem_1.py#L70)
- Task 激活需全部前驱 Task 完成；同核前一 Task 完成后加同核等待；跨核前驱完成后加跨核等待。所有约束取最大 release time，而非逐个串加。[激活规则](2026_official/code/multicore_cut_evaluate_problem_1.py#L318)
- 每个 Task 独立经过核内 Step1/2/3；容量不足由 Step2 插入 spill。全局模拟使用每核固定 Pipe 顺序，并共享 DDR 带宽。[评估主流程](2026_official/code/multicore_cut_evaluate_problem_1.py#L200)

## Problem 2：场景 B

- 同核全部子图合并为一个 Task；同核 tensor 可驻留复用。图输入按消费核心各读一次，最终输出按生产核心写回；中间 tensor 对每个实际 source-core/target-core 组合插入跨核 COPY 对。多个消费者通过 core 集合归并，不简单按 edge 扇出重复复制。同核依赖不产生边界 COPY。[每核 Task 与 tensor 传输](2026_official/code/multicore_cut_evaluate_problem_2.py#L61)
- 每条跨核传输独立关联源 `COPY_OUT` 与目标 `COPY_IN`。目标 COPY 在源写回完成并经过配置延迟后释放，不要求源核心整个 Task 完成；多前驱目标按全部 release 条件满足后就绪。[跨核 COPY release](2026_official/code/multicore_cut_evaluate_problem_2.py#L374)
- 全局校验包含本地执行依赖、每 Pipe FIFO 顺序及跨核 COPY 依赖，因此核间数据往返允许存在，但组合后的实际操作等待图不能成环。[执行图校验](2026_official/code/evaluation_validation.py#L261)

## Problem 3：只读 FIFO L2

- Task/跨核结构沿用 Problem 2。只有 `COPY_IN` 查询 Cache；查询发生在操作发射时，key 为输出片上 tensor 的 `logical_tid`（无该字段时退回 tensor ID）。命中使用独立 `CACHE_READ` 带宽池，不占 DDR；命中不刷新 FIFO。未命中使用 DDR，并在 COPY_IN 完成后插入 Cache；插入按 FIFO 淘汰。大于 Cache 容量的 tensor 不插入；并发首次 miss 不合并。[Cache key/插入/查询](2026_official/code/multicore_cut_evaluate_problem_3.py#L398) [发射与带宽池](2026_official/code/multicore_cut_evaluate_problem_3.py#L539)
- DDR 与 Cache Read 是独立共享带宽池，各自公平分配剩余工作量。最终 `cache_stats.hit_rate` 按命中字节/可缓存访问总字节计算，不是命中次数比率。[带宽池与统计](2026_official/code/multicore_cut_evaluate_problem_3.py#L322) [命中率](2026_official/code/multicore_cut_evaluate_problem_3.py#L676)

## Spill、Pipe 与共享 DDR

- Step1 对 op/tensor 图桥接出 op 依赖并生成确定性拓扑访问序。Step2 依据 L1/UB 分别维护活跃 tensor；检查点是输出分配后、当前操作末次输入释放前。容量超限时排除本操作正在使用和没有未来 use 的 tensor，选择下一次使用最晚者 spill。首次 spill 写入 DDR backing，后续复用 backing；换入生成新物理 tensor incarnation，并沿逻辑 tensor ID 关联。[Step1](2026_official/code/schedule_step1.py#L106) [Step2](2026_official/code/schedule_step2.py#L144)
- Step3 通过 tensor 引用计数和虚拟容量块控制 alloc/free；复用释放的容量前添加 WAR/WAW op 依赖并检查死锁/环。L1 与 UB 是分别容量，不是统一内存池。[内存复用依赖](2026_official/code/schedule_step3.py#L144)
- 最终多核模拟在搬运加入/完成时结算剩余工作并更新同一 DDR 池中的在途 COPY；各搬运公平分享总 DDR 带宽。Step3 日志的临时 `base_delta`/`adjusted_delta` 展开后，最终每层完成时间为 `delta_work * active_count`，与公平共享一致，不应把日志中的“penalty”误读为另一个最终带宽规则。[多核 DDR 模拟](2026_official/code/multicore_cut_evaluate_problem_1.py#L235) [Step3 投影](2026_official/code/schedule_step3.py#L342)

## CLI 与真实输出字段

三个评估器命令均接收 `graph [plan] --config CONFIG -o RESULT --trace-output TRACE --log-output LOG`；省略 plan 时读取 `<graph stem>_multicore_res.json`，省略 config 时读取图同目录 `config.txt`。无方案生成器不等于优化基线。[CLI](2026_official/code/contest_io.py#L181)

成功结果共同包含 `scene`、`makespan`、`num_cores`、`bandwidth_bytes_per_cycle`、`capacity_bytes`、`memory_peak_by_core`、`cross_task_traffic`、`data_movement_bytes`、`per_core_timeline`。Problem 1 增加 `step3_by_task`、两个 Task wait 值、`task_dependencies`、`ddr_contention_log`。Problem 2 增加 `step3_by_core`、跨核延迟、`task_count`、`task_dependencies`、`cross_core_transfers`。Problem 3 还增加 `problem=3`、`cache_mode`、Cache 容量/带宽、`cache_stats`、`cache_events`、`cache_final_entries`、`cache_used_bytes_final`；其 `scene` 字段仍为 `B`。[Problem 1 字段](2026_official/code/multicore_cut_evaluate_problem_1.py#L500) [Problem 2 字段](2026_official/code/multicore_cut_evaluate_problem_2.py#L573) [Problem 3 字段](2026_official/code/multicore_cut_evaluate_problem_3.py#L684)

`data_movement_bytes` 含 `original_graph_copy_bytes`、`scheduled_copy_bytes`、`added_copy_bytes`、`partition_added_copy_bytes`、`spill_added_copy_bytes`；Problem 3 的逻辑 COPY 量不应误作实际 DDR 命中字节，Cache 命中另由 `cache_stats.hit_bytes/miss_bytes` 报告。

## 与此前假设的差异及核验

1. 题面概述只强调 Tensor/Op 边；官方 validator 还接受直接 Op→Op 边，Problem 2/3 会处理其 `data_size`。当前 `scheduler.py` 仍拒绝这种边，必须修正。
2. 官方 `cycles` 可为0，但核内 duration 对普通 op 至少按1 cycle；零周期不能据此推断零耗时。
3. Result JSON 不含单一的 `cache_hit_rate` 顶层字段；命中字节率位于 `cache_stats.hit_rate`。
4. 只读 Cache 查询在发射时发生、未命中完成后插入；同一 logical tensor 的同时首次 miss 不会自动合并。
5. 题面手工 4-op 示例（260 cycles）与验收最小图（1个 ADD、两个16-byte COPY，总 Makespan 6）不同。验收最小图已对三份官方评估器实际运行通过：三题均 `makespan=6`、原始/调度 COPY 量32 bytes、新增量0；Problem 3 `hit_rate=0.0`。
