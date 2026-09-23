# 第一版实现报告

## 范围与真实性

已将用户提供的 2026 附件 ZIP 解压到 `2026_official/`，原压缩包保持不变。实现只采用当前题目的 `tensors/ops/edges` 数据和两字段多核方案，不使用 2025 年的 ALLOC/FREE、L0 缓冲区或 SPILL 输出格式。正式 Makespan、搬运量、Cache 命中率均来自官方 Python evaluator；图划分代理值不作为结果。

本报告记录第一轮工程闭环，不代表已完成 100 个用例的最终提交实验。100 个官方 JSON 均通过官方 `validate_graph`；完整三问题实验目前只覆盖部分代表用例，未评估组合明确视为缺失。

## 文件结构

```text
2026_official/                  官方附件解压目录
solver/
  graph_io.py                    官方图与配置读取
  graph_analysis.py              op DAG、COPY 收缩、拓扑和路径分析
  features.py                    操作/子图、Pipe、tensor 生存期特征
  partition.py                   连续拓扑切分候选
  legality.py                    官方方案推导与合法性校验
  schedule_common.py             关键路径列表分核与场景代价
  schedule_a.py                  Problem 1 入口
  schedule_b.py                  Problem 2 入口
  cache_aware.py                 Problem 3/FIFO 代理入口
  evaluate_adapter.py            官方 evaluator 子进程适配
  improve.py                     受限迁移、合并邻域
  solver.py                      时间/次数预算、best-so-far 与 CLI
experiments/
  profile_data.py                全部正式图画像
  run_all.py                     批量运行与 resume
  collect_results.py             正式结果重算及平均加速比
  make_figures.py                纯标准库 SVG 图生成
 tests/
  fixtures/                      官方最小图与方案
  test_legality.py               非法切图/顺序等检查
  test_small_graphs.py           合成 DAG 和调度器测试
  test_problem_semantics.py       官方 evaluator 微图语义测试
results/
  raw/ schedules/ traces/ logs/   评估产物
  summary.csv summary.json       逐组合结果
  aggregate_summary.csv/json     平均曲线数据
  figures/                       自动生成的 SVG
```

## 官方语义核对

细节和源代码位置见 [docs_understanding.md](docs_understanding.md)。核实后的实现依据包括：Problem 1 每子图一个 Task、同核跨 Task 仍经 DDR；Problem 2/3 每核合并 Task、跨核目标 COPY_IN 等源 COPY_OUT 加延迟；L1/UB 分开容量并由 Step2 spill；四条 Pipe 每核每条一个在途槽位；全局 DDR 公平共享；Problem 3 在 COPY_IN 发射时查逻辑 tensor ID，命中不刷新 FIFO，miss 完成后插入，Cache 和 DDR 带宽独立。

验收最小图已实际运行三份 evaluator，均返回 Makespan=6 cycles、`original_graph_copy_bytes=32`、`scheduled_copy_bytes=32`、`added_copy_bytes=0`；Problem 3 的 Cache hit rate 为0。

## 算法实现

- 通过官方 validator 检查原图；以 op/tensor 索引构建邻接和张量消费者。COPY 节点收缩使用一次 op DAG 拓扑传播，得到计算 op 间最近依赖。
- 计算拓扑层、前/后向最长周期、Pipe M/V 工作量、外部 DDR 输入、tensor size/fan-out 与拓扑位置生存期。
- 保证合法的初始解采用全局拓扑序的连续区间分块；Problem 1/2/3 候选粒度分别使用少量不同的核数倍数。
- 子图排序以 upward rank 和 ready 列表为主；分核估计 M/V/搬运 Pipe 负载、边界 tensor、跨核同步。Problem 1 加 Task 切换等待和 DDR 边界；Problem 2 聚合同核数据；Problem 3 使用有容量限制、命中不刷新 FIFO 的 Cache 近似状态。所有预测只决定候选排序，正式比较仍由官方 evaluator 给出。
- 每个候选先过官方 `derive_multicore_plan`，通过后才写入独立 JSON 并调用 evaluator。evaluator 调用有超时和最大次数；最好合法方案在评估前落盘。局部迁移/相邻子图合并邻域已实现，但首轮正式矩阵用 `--no-improve` 控制 evaluator 预算，尚未测出局部搜索收益。

## 复杂度与资源

令 $V$ 为图节点/op 数，$E$ 为边数，$G$ 为候选子图数，$N\leq5$ 为核心数。JSON 解析、索引和基础拓扑为 $O(V+E)$（拓扑堆实现带有对数因子）；子图特征的实现为 $O(G(V+E))$ 的保守上界，且当前核数范围把候选 $G$ 限制在不超过 $6N$ 的小常数倍；列表分核为 $O(GN\log G)$ 加通信特征处理。空间为 $O(V+E+T)$，不构造传递闭包或消费者完全图。官方核内调度和多核事件模拟时间由官方实现决定，并不计入这些代理复杂度。

运行依赖 Python 3.11 标准库及官方附件中的代码，无新增第三方依赖。Matplotlib 在所选环境中因未安装 pip 无法安装，因此采用标准库直接输出 SVG，而未留下不可用依赖。

## 数据画像

扫描并通过官方图校验的正式用例数为100。`data_profile.csv` 和 `data_profile.md` 是可复现输出。非 COPY 操作数范围552–35,705，中位数3,720.5；最小图为 `case_093`（552 个非 COPY op），选作中位规模代表的 `case_034` 有3,745个，最大图 `case_014` 有35,705个。

## 已运行测试

`python -m unittest discover -s tests -v`：13项全部通过。覆盖官方方案映射和子图环、空核心、链与分叉图、同核 Problem 1/2 COPY 差异、core0→core1→core0、两核心同时从 DDR 读时各自使用共享带宽、首次同时 Cache miss、FIFO 命中不刷新、超容量 tensor 不缓存。

模块化 CLI 的官方最小图三问通过。第一轮选定图中，`case_019`、`case_093`、`case_034` 的 Problem 1/2/3 在2–5核全部得到 evaluator 结果；另补齐这些图 Problem 2/3 的1核结果。`case_001` 也运行了三问题的2核版本。

最后修正 Problem 2/3 代理中跨核源 COPY_OUT release 估算后，再次运行13项单元/集成测试，全部通过；并使用最终代码重跑 `case_019` 的 Problem 2/3、2核，官方结果分别为 62,550 和79,137 cycles。

## 首轮结果

下表的加速比均以官方单核固定基准为分母，跨用例汇总取逐用例比值算术平均。由于仅5个 case 中有部分完整数据，表中报告实际纳入数，不代表100用例最终平均。

| 指标 | 核数 | 算术平均 | 已评用例/正式用例 |
|---|---:|---:|---:|
| Problem 1 加速比 | 1 | 1.000 | 4/5 |
| Problem 1 加速比 | 2 | 1.494 | 4/5 |
| Problem 1 加速比 | 3 | 1.824 | 3/5 |
| Problem 1 加速比 | 4 | 2.430 | 3/5 |
| Problem 1 加速比 | 5 | 2.906 | 3/5 |
| Problem 2 加速比 | 1 | 1.000 | 3/5 |
| Problem 2 加速比 | 2 | 1.644 | 4/5 |
| Problem 2 加速比 | 3 | 2.122 | 3/5 |
| Problem 2 加速比 | 4 | 2.560 | 3/5 |
| Problem 2 加速比 | 5 | 2.923 | 3/5 |
| Problem 3 无 L2 加速比 | 2 | 1.644 | 4/5 |
| Problem 3 有 L2 加速比 | 2 | 1.576 | 4/5 |
| Problem 3 无 L2 加速比 | 1 | 1.000 | 3/5 |
| Problem 3 有 L2 加速比 | 1 | 1.000 | 3/5 |
| Problem 3 无 L2 加速比 | 3 | 2.122 | 3/5 |
| Problem 3 有 L2 加速比 | 3 | 1.898 | 3/5 |
| Problem 3 无 L2 加速比 | 4 | 2.560 | 3/5 |
| Problem 3 有 L2 加速比 | 4 | 2.134 | 3/5 |
| Problem 3 无 L2 加速比 | 5 | 2.923 | 3/5 |
| Problem 3 有 L2 加速比 | 5 | 2.658 | 3/5 |
| Problem 3 Cache speedup | 2 | 0.983 | 5/5 配对 |
| Problem 3 Cache speedup | 3 | 0.875 | 4/5 配对 |
| Problem 3 Cache speedup | 4 | 0.779 | 4/5 配对 |
| Problem 3 Cache speedup | 5 | 0.889 | 4/5 配对 |

2核逐用例结果（`Makespan / added_copy_bytes`）：

| 用例 | Problem 1 | Problem 2 | Problem 3 | Problem 3 Cache hit rate |
|---|---:|---:|---:|---:|
| `case_001` | 116,868 / 1,152 | 116,868 / 1,152 | 116,868 / 1,152 | 0.000 |
| `case_019` | 79,348 / 64 | 62,550 / 36 | 79,137 / 0 | 0.000 |
| `case_034` | 846,983 / 46,860 | 637,987 / 614,684 | 638,004 / 679,196 | 0.173 |
| `case_093` | 37,827 / 8,192 | 37,827 / 8,192 | 37,827 / 8,192 | 0.000 |
| `case_014` | 18,184,664 / 65,626,724 | 14,977,860 / 70,362,528 | 13,321,252 / 75,050,400 | 0.185 |

最大图 `case_014` 四核数组合（Makespan cycles）：

| 核数 | Problem 1 | Problem 2 | Problem 3 |
|---:|---:|---:|---:|
| 2 | 18,184,664 | 14,977,860 | 13,321,252 |
| 3 | 18,206,061 | 10,253,458 | 11,874,509 |
| 4 | 13,624,358 | 8,636,778 | 11,145,703 |
| 5 | 13,630,764 | 6,969,885 | 7,099,894 |

以上 `case_014` 结果均缺少固定单核基准，故不计算 speedup；该图 Problem 1 的 3 核略慢于2核，反映本版切分和列表分核还不稳定。

L2 首轮结果多数未改善 Makespan：这说明当前 cache-aware 候选和列表排序不足以充分错开共享输入的读取时机，不能据命中率声称 L2 优化成功。个别用例有命中（例如 `case_034`），但命中字节率不是 Makespan 改善的充分条件。

最大图 `case_014` 的 Problem 1 单核基准 evaluator 在300秒预算内超时。独立基准预算修复后，Problem 1/2/3 在2–5核均得到官方评估结果；最大图多核候选 evaluator 用时约16–161秒。2核 Problem 1 的 Makespan 为18,184,664 cycles、added COPY 为65,626,724 bytes；因单核基准不可用不报告加速比。首轮最大图高 COPY/大 Makespan 表明当前切块粒度在该图上很差，应优先改进，而不应包装为成功优化。

## 评估调用与运行时间

到本报告数据汇总时，`summary.json` 有57行，当前索引合计66次 evaluator 调用，57个组合状态均为 evaluated。统计只累加当前保留的每组合行；重试前被覆盖的旧调用不计入此和。三个代表图各自包含单核基准与三场景2–5核，另含 Problem 2/3 单核；标准核心组合的单次调度求解一般在亚秒级，官方 evaluator 用时随图规模增长。`case_014` 单核基准300秒超时；最大图官方多核运行约16–161秒，取决于场景/核心数。精确每组合秒数、调用数和错误保存在 `results/summary.json` / `summary.csv`。

## 当前瓶颈

1. 连续区间切图合法稳健，但不能把跨分支的大 tensor 消费者共同聚合，Problem 2/3 通信代理对同核候选仍较粗。
2. 当前只试有限粒度和一份列表调度候选，复杂图的 Pareto 选择和 MIG/合并局部改进未系统搜索。
3. Problem 3 Cache 状态是候选生成期的代理，而非官方事件模拟的完整调度反馈；从首轮配对比可见 Cache 反而减速。
4. `case_014` 单核基准官方运行超过300秒，需独立评估大图核内基准耗时并调整实验预算。
5. 目前正式实验只覆盖5/100个 case，禁止将当前均值作为最终竞赛平均曲线。

## 失败尝试与修正

- 最初旧版单文件调度代理没有官方 evaluator 接口，已替换为模块化 solver + 官方 adapter；旧版不用于实验结果。
- 一个同核通信测试 fixture 起初把相邻子图放在不同核，导致断言不符合测试目的；fixture 修正后，A 场景新增32 bytes、B 场景新增0 bytes，官方测试通过。
- `case_014` 单核固定基准在300秒超时，初始逻辑让其耗尽每组合预算；已将基准超时独立化、保留失败标记，并让候选仍可在单独预算内评估。
- 无法安装 Matplotlib，因为选定环境缺 pip；切换为纯标准库 SVG，不保留失效依赖。

## 最值得做的后续实验

1. 针对 `case_014` 对照4、8、16、32个拓扑区间，找出搬运暴涨的粒度阈值，并在统一核内基准上比较。
2. 对 `case_019`、`case_034` 做 baseline 与有限迁移/合并邻域的对照，记录每次局部候选的 makespan 和 added COPY，确认邻域是否有效。
3. 对 `case_034`、`case_019` 的共享输入分别比较同核聚合、跨核 fan-out 和 FIFO 插入时序，解释 Problem 3 的命中却减速。
4. 用已选出的5个代表图扩至全部 `case_001`–`case_100`，先每 case 每场景每核最多1–3次 evaluator 调用，之后仅在可改进 case 增加预算。
5. 单独评估官方 `singlecore_evaluate.py` 在最大规模图的耗时；若超过竞赛时间预算，确保基准由官方固定入口在受控批次产生，而不复用新的多核优化结果替代。
