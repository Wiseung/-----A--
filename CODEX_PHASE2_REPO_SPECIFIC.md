# Codex 第二轮任务：基于实际仓库审查的增量修复与实验

## 0. 范围与版本

目标仓库：Wiseung/-----A--。
本任务书审查快照：main，30a62dd28cb69d475ca7862fe405aa1e6a493711，提交说明 first commit。
任务：继续优化 2026 多核切图调度工程，不重建已经存在的工程，不照搬 2025 核内题。

本任务书依据该提交的源代码、已有 summary、方案、原始结果片段和日志制定。撰写者进行了静态代码审查和产物交叉核对，没有重新运行官方完整评估；下面的性能数字均来自仓库原有产物。你执行时需要实际重跑，不能把任务书数字当作新实验数字。

如果当前工作区已前进：先与审查快照比对，只修仍然存在的问题。尊重未提交修改。只进行用户授权的本地修改和测试，不擅自 push、创建 PR、改远端分支或清空 results。

保留原 results；新实验使用 results_round2/<experiment_id>/<run_id>/。正式评估不得修改 2026_official/code 或 data/config.txt。先用根目录原附件 ZIP 核对解压官方文件，报告真实差异，不自行假定一致。

## 1. 已经完成的部分，应复用

- solver/ 已有图解析、COPY 收缩、连续拓扑分块、统一列表分核、场景入口、官方评估适配、有限迁移/合并邻域、CLI。
- experiments/ 已有画像、批量运行、汇总和标准库 SVG 输出。
- tests/ 已有 13 项测试；报告称通过，执行时重跑验证。
- 官方 validator 已被图解析和方案校验调用，且评估器已使用 subprocess 参数数组、sys.executable、独立临时目录。
- 正式数据报告为 100 个 case；非 COPY 操作 552–35,705。不要继续把“定位附件、建立目录”作为主要开发工作。

## 2. 当前有证据的性能问题

| 用例/核数 | 现有问题2 | 现有问题3 | 说明 |
|---|---:|---:|---|
| case_019 / 4核 | 42158 | 78089 | p3 只使用核心0，另外3核为空，命中率0 |
| case_034 / 4核 | 420981 | 525396 | p3 命中率约9.23%，仍更慢 |
| case_014 / 4核 | 8636778 | 11145703 | p3 核心0结束最晚，其余核心约230万周期已结束 |

另外，case_014 问题1两核方案 core_schedules=[[0,1],[]]，日志显示只有核心0工作，Makespan=18184664。
case_093 是保留性能的对照：问题2五核15876，问题3五核15630，正式单核参考74205。

这些比较采用不同方案，不能宣称“相同调度开启 Cache 一定变慢”。必须做固定方案交叉评估。

源文件：
- results/summary.csv / summary.json
- results/logs/case_019_p2_n4_eval_0001_g8.txt
- results/logs/case_019_p3_n4_eval_0001_g8.txt
- results/schedules/case_019_p2_n4_best.json
- results/schedules/case_019_p3_n4_best.json
- results/logs/case_014_p1_n2_eval_0001_g2.txt
- results/logs/case_014_p2_n4_eval_0001_g8.txt
- results/logs/case_014_p3_n4_eval_0001_g8.txt

## 3. P0：先修具体错误，再加算法

### P0-1：完成时间在候选提交时被丢失

位置：solver/schedule_common.py，schedule_partition 的 problem != 1 分支和最终提交部分。

现有逻辑先算：

    finish = max(start + internal_critical_path, max(pipe_ends.values()))

但 best/candidate 不保存 finish，最后：

    estimated_finish[group_id] = max(pipe_ends.values())

会丢失关键路径限制。例如同一子图 M(80) -> V(80)，CP=160，两条 Pipe 工作量分别80，score中finish160，提交状态却可能记录80。

修复要求：
1. 用明确的 PlacementEstimate/dataclass 或等价结构同时保存 score、group_finish、各 Pipe 结束时刻、输出可用时刻和通信估计。
2. 评分和提交使用同一估计，不能重建较小的完成时间。
3. 增加 CP 大于单 Pipe 累计时间的回归测试。
4. 不把这项修复宣称为精确模拟；下述读写与依赖建模仍需处理。

### P0-2：场景B/问题3输出搬运资源未完整进入代价

位置：schedule_common.py 的 output_copy_bytes、group_work、source_ready。

当前 output_copy_bytes 仅在问题1加入 PIPE_MTE3。B/3虽然给跨核父子依赖加一个字节/带宽延迟，但没有完整建模源核写出排队、扇出多次写出、最终输出写回以及源核搬运 Pipe 占用。

要求：
- 分开维护源核 MTE3 写出工作、目标核 MTE2 读入工作、固定同步延迟。
- 显式保留最终输出。
- 增量通信按官方构造规则计算，不能只添加无资源归属的 edge latency。
- 读入和依赖计算不能无条件同时从 source_ready 开始。先建立轻量局部依赖排布，使输入可用时刻影响消费者；不是重写官方全局模拟器。

### P0-3：上一轮通信近似不能原样套用场景B

实际源码：2026_official/code/multicore_cut_evaluate_problem_2.py，_build_scene_b_tasks。

对中间 tensor，每个 source_core -> target_core 连接分别调用 add_copy_out 和 add_copy_in。单生产核、r个外部消费者核时，仅跨核连接对应的搬运小计为 2*r*size，不是 (1+r)*size。最终输出和原始输入另行计算，多生产核情况直接遵循源码。

问题1则按每个Task的输入/输出边界处理，不能用B的公式替换A。

要求：
- 用单张量跨1/2/3个目标核、同目标核多个消费者、最终输出兼跨核等微图，与官方 partition_added_copy_bytes 逐项对照。
- 建立 TensorTransferPlan 或同等表示，保留生产者/消费者集合、各核使用计数、COPY对集合。
- 不改官方评估器来配合代理。

### P0-4：候选核心的第二排序键目前是常量

位置：schedule_common.py，B/3分支：

    extra_bytes = sum(inputs[key][1] for key in inputs)
    score = (finish, float(extra_bytes), core_id)

对同一 group，extra_bytes 不随候选 core 变化，因此该排序键没有表达差分通信。部分通信已通过读入周期参与第一排序键，不能说代码完全不考虑通信；这里要修的是差分第二键及相关日志。

要求：记录 delta_cross_copy_bytes、delta_repeated_input_bytes、delta_write_work、delta_read_work，按候选核心计算真实变化。

### P0-5：计时错误与超时参数

位置：solver/solver.py，solve_one。

单核基准在 solve_started 之前运行，但成功分支 solver_runtime_sec 从 total_elapsed 又扣除了包含 baseline_runtime 的 evaluator_runtime，重复扣除基准时间，并被 max(0,...) 掩盖。

要求：分别记录 baseline_wall_sec、search_wall_sec、candidate_eval_wall_sec、solver_cpu_or_overhead_sec、total_wall_sec。修正成功与失败分支一致性。单次候选超时应受 min(remaining_budget, per_eval_timeout) 限制，而不是覆盖 --evaluator-timeout。报告冷启动和缓存复用两种成本。

### P0-6：历史最好方案与运行产物不可覆盖

位置：solver.py 的 try_candidate/_store_summary；evaluate_adapter.py 的 tag 归档；run_all.py 的 resume。

当前第一次候选会写覆盖 *_best.json，即便尚未评估；每次进程 evaluator.calls 归零，归档tag可能跨运行重名；summary按(case,problem,ncores)覆盖，不区分代码、方法、seed和预算；baseline按case stem直接复用。

要求：
- legal_draft 与 best_validated 分开。
- 新运行首先加载匹配哈希的历史有效方案，任何失败不得破坏它。
- run_id + candidate_id 唯一保存每次方案、原始结果、日志及失败。
- 扩展运行身份：graph/config/official/solver哈希、problem、ncores、method、seed、预算。
- exact evaluation cache至少包含graph、plan、problem、config、official代码哈希。
- resume只能复用同一实验身份，不静默跳过新版本实验。
- 单进程汇总或独立结果文件，禁止并行读改写同一个summary。

## 4. P1：先改候选管理，防止问题3退化

当前 candidate_group_counts：A=(N,2N,4N)，B=(2N,4N)，B+L2=(2N,4N,6N)。solve_one按列表顺序逐个评估，当前多数记录max-evals=1且no-improve，所以实际基本只评首个粒度。这里不是已经存在一个全局代理模型把方案筛掉，而是候选集没有得到充分评估。

新的小型候选集合：
1. 当前快照同场景已验证方案（作为旧版对照，不伪装免费冷启动）。
2. N-1核最好方案追加空核，在当前场景重新确认。
3. 当前场景构造器的少量粒度。
4. B方案在B+L2中的真实评估（问题3必须加入）。
5. 结构感知划分的一到两个候选。
6. 必要时整图单Task保底；仍使用当前场景评估，不能拿单核参考数字直接替代。

必须给候选家族保留有限配额，预留局部改进预算。完整记录候选来源和拒绝原因。所有比较以官方Makespan、added_copy_bytes按字典序为准。不能强制全部核心非空。

## 5. P1：Cache 代理改为因果一致的诊断/后处理

位置：schedule_common.py 的 cache、_update_cache、cache_misses。

当前构造完一个子图就立即把未来才完成的miss写入cache并可能淘汰旧项，只用available_at阻止过早hit。这仍会让未来插入影响过去查询；cache_misses集合遍历也不是实际完成时间顺序。贪心分配顺序不等于模拟时间顺序。

执行顺序：
1. 保留原Cache构造器作为待比较候选，但不让它成为问题3唯一入口。
2. 首先使用B优质方案在问题3评估。
3. 从官方cache_events读取hit/miss/insert/eviction，保持原事件顺序，结合真实时间和跨核传输诊断。
4. 区分首次miss、在途重复miss、淘汰后miss；证据不足时标unknown。
5. 初期仅在固定划分/分核下改变合法子图顺序，逐次官方评估。
6. 若实现时间代理，用完成事件堆或完整访问事件重放，所有query仅看到其发射时刻前已完成的插入。不能只给立即写入的字典附一个future timestamp。
7. 逻辑tensor ID严格对齐官方copy_tensor_info；不要擅自换成DDR源ID。
8. 不添加任何JSON显式delay/start_time字段。

## 6. P1：结构感知划分，而不是只调切块数

位置：graph_analysis.py、partition.py、features.py、schedule_common.py。

现有拓扑序由最小ID优先的Kahn堆生成，划分仅在这一序列上切连续区间。这个框架应保留为保底，但不应作为唯一方案空间。

新增候选优先顺序：
- 不改变节点ID的多种合法拓扑序：依赖分支局部性、关键路径、预计释放字节。
- 弱耦合分支或依赖链聚合；共享输入作为带权超边处理，不把消费者构成完全图。
- 带容量风险和工作量边界的区域增长/安全合并。
- 每次收缩后校验子图DAG和场景对应执行合法性。

必要的配套修改：
- schedule_partition中rank必须沿真正的子图逆拓扑序，不依赖sgid降序。
- partition_contiguous当前平衡项除以全图total，cut项除以max_cut；子图数增加时相对权重改变。新增局部目标工作量归一化及合法切点窗口；不要承诺某个固定权重最优。
- 单次逆拓扑扫描计算各组内部CP；避免每个group扫描全图。
- 静态operation_features和cut profile复用。
- graph_features里的shared_inputs嵌套扫描改为一次计数。

## 7. P1：真正接入缓存生存期

当前features.py已有l1_pressure_est、ub_pressure_est、residence_cost_est，但schedule_common.py没有把这些字段用于核心评分。不要在论文中说已实现了有效的驻留惩罚。

先固定划分和核心映射，仅研究顺序：大tensor尽早完成全部消费者、非关键生产者适度延后、独立子图交换/插入。随后才研究迁核与重排组合。

场景B按整核序列重算生存期，不能把单子图内峰值简单相加。粗略生存期指标只能作为风险，不能当作官方判非法的依据。同核拆分子图不会自动清空缓存。

使用官方现成data_movement_bytes.partition_added_copy_bytes与spill_added_copy_bytes定位大图的搬运来源。

## 8. P1：修复局部邻域的顺序破坏与预算饥饿

位置：improve.py。

当前_plan_with_owners把每核sgid全部数值排序，迁移一个group可能同时改变多个原有顺序。当前generate_neighbors先遍历全部迁移，再处理A合并；小limit可能使后者根本没有机会。现在没有split、swap、重插入或Cache专用邻域。

要求：
- 迁移只删除源组并插入目标核的合法位置，保持其余组相对顺序。
- 分离ownership move和order move，让实验可解释。
- 以瓶颈张量、关键链、过载核心选少量group，不优先扫最小sgid。
- 各邻域配额或轮转，避免全部预算耗在迁移。
- 增加独立组swap/reinsert、瓶颈group split、A安全merge。
- P1检查Task依赖+同核顺序；B/3不能套成整核Task屏障，最终由官方实际展开图检查。

## 9. P0/P1：实验汇总与日志口径

collect_results.py当前cases来自summary已出现的case，因此只计算5个已出现用例上的missing，而不是100个正式用例上的missing。正式矩阵必须来自case_manifest。

主多核组合100*3*4=1200。当前57行包括51个2–5核组合与6个单核B/L2组合，主多核覆盖约4.25%，不能称为全量结果。

要求：
- 全量矩阵显式记录not_run/timeout/invalid/evaluation_failed/success。
- 单核参考缺失与多核评估失败分开统计。
- 分子分母按相同case配对，曲线标明覆盖数和共同case集合。
- 旧版、修复版、优化版分开保存；累计调用数来自事件账本，不来自最后一行覆盖后的求和。
- Cache结果分别报告固定方案硬件效果、算法适配效果和二者合计效果。

特别警告：2026_official/code/contest_io.py的format_data_movement_log在physical_ddr_bytes缺失时回退scheduled_copy_bytes。问题3日志里即使命中很多，这两个数仍可能一样。不要把该显示值直接当物理DDR流量，也不要改官方代码；外部诊断需核对实际COPY种类、命中字节及计数字段后另算physical_ddr_bytes_derived。

## 10. 立即可以执行的两个隔离实验

### R00：固定方案2×2对照

对象优先case_019四核，再case_034/014。
方案xB=现有问题2方案；方案xC=现有问题3方案。
分别计算T_B(xB), T_C(xB), T_B(xC), T_C(xC)。

报告：
- 硬件效果 T_B(xB)/T_C(xB)
- Cache下算法适配效果 T_C(xB)/T_C(xC)
- 综合效果 T_B(xB)/T_C(xC)

不要预填交叉评估数字。

### R01：划分策略与评分策略分离

case_019，n=4，g=8，固定使用问题3官方评估。
生成四种组合：
- partition(problem=2), schedule(problem=2)
- partition(problem=2), schedule(problem=3)
- partition(problem=3), schedule(problem=2)
- partition(problem=3), schedule(problem=3)

分开解释问题3切分cut_weight变化与Cache选核代理的影响。保存每个group的逐core评分、最终active_core_count、M/V负载、官方Makespan。

## 11. 后续实验矩阵

| 编号 | 主要变化 | 目的 |
|---|---|---|
| R02 | 只修finish一致性/计时/运行身份 | 区分正确性修复与算法收益 |
| R03 | 加B→L2、少核扩展、旧版保底候选 | 防止候选退化 |
| R04 | 补源MTE3、真实差分通信 | 校准B/3选核 |
| R05 | 多拓扑序、分支感知划分 | 释放被分块限制的并行性 |
| R06 | 局部切点平衡与粒度 | 消除过大/过小子图 |
| R07 | 固定分核的生存期重排 | 降低spill和长驻留 |
| R08 | 时序一致的Cache后处理 | 改善有价值的复用 |
| R09 | 保序迁移/交换/拆分/合并配额 | 验证有限局部改进 |
| R10 | 固定次数与固定墙钟双预算 | 公平比较求解效率 |
| R11 | 分层8-case回归 | 检查泛化、避免只调3个图 |
| R12 | 100-case全量覆盖及消融 | 形成正式论文结果 |

建议诊断集：case_001、014、016、019、020、028、034、093。
理由来自data_profile：001与093为已有良好对照；019有明显分核退化；034有spill/Cache；014非COPY节点数最大；016向量独占且依赖深；020向量宽图；028矩阵计算量更大、不能只用节点数选“大图”。

最先在019/034/093的2核、4核上验证，再处理014及新增图。超参数不得以case名硬编码。候选数和预算都是实验设置，不是题面新增约束。

## 12. 单核基准与预算

case_014单核基准已在原环境300秒超时，这是缺失，不是无解。单独profiling官方入口，记录Step1/2/3和输出时间；可在受控离线实验给更长预算，不能拿优化后单核或多核数字替代。

每(problem,ncore)搜索预算与整个case全部场景总墙钟时间分别记录。固定评估次数用于比较候选质量；固定墙钟用于比较部署成本。历史方案、缓存及单核参考的生成成本应透明披露。不要声称每个组合5分钟就意味着整个case5分钟。

不重写官方模拟器来追求速度。先减少无效候选、静态特征重复计算、日志I/O。需要进程内调用或缓存Task展开时，另开实验并验证完整上下文依赖与官方原入口等价。

## 13. 新增测试

至少包含：
- CP大于max单Pipe工作量，score与提交finish一致。
- 同tensor跨多个目标核，A/B通信计数分别匹配官方。
- 输入可用之前不能启动依赖计算的轻量代理约束。
- 未来Cache插入不能淘汰过去时刻仍在Cache中的对象。
- 候选核通信差分确实随core改变。
- 迁移一个group时其他group相对顺序不变。
- 小邻域预算也能覆盖多类操作。
- 失败候选不覆盖历史best_validated。
- 同case不同代码/配置/预算不能错误resume。
- 单核基准耗时不重复扣减。
- manifest有100case而summary仅5case时，缺失应包含其余95case。
- 非拓扑sgid编号下rank仍正确。
- 旧有13项测试全部重跑。

语义微测试可以显式使用小Cache等人工配置，但必须标注synthetic，不进入正式固定配置结果。不要把这种测试配置变化误报为比赛实验改参。

## 14. 交付与验收

建议分五个小步：
A. 版本冻结、运行账本、计时、finish修复。
B. 固定方案对照与候选保底。
C. 真实通信、输出时间、结构划分。
D. 生存期/Cache/局部邻域。
E. 分层到全量实验。

每步留下：真实改动、命令、测试结果、成功/失败、单因素性能对照、运行成本。

交付文件至少包括：
- REPO_AUDIT_ROUND2.md
- FIX_REPORT.md
- candidate_trials.jsonl（或等价不可变记录）
- fixed_plan_cache_comparison.json
- subgraph_diagnostics.json
- summary_by_run.json
- complete_case_manifest.json
- 实际可执行的新增测试与实验脚本
- ROUND2_RESULTS.md

每个性能数字必须追溯到方案、输入、官方原始结果和版本。优先保证实现与证据一致，不要求虚构某个百分比提升。若新策略失败，保留并报告负结果，最终返回当前场景下真正评估过的最好合法方案。
