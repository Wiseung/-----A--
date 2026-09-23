# 2026 多核调度求解器

此工程读取 2026 官方 `tensors/ops/edges` 计算图，只输出赛题规定的 `node_to_subgraph` 和 `core_schedules`。实际 Makespan、数据搬运与 Cache 指标均由 `2026_official/code/` 官方评估器计算。

## 目录

- `2026_official/`：原附件 ZIP 的原样解压目录；ZIP 保留在工作区根目录。
- `solver/`：图解析、依赖分析、特征、连续拓扑分块、合法性、场景调度、评估适配器和有界局部邻域。
- `experiments/`：正式数据画像、批量运行、结果汇总和 SVG 图生成。
- `tests/`：标准库 unittest 微型图和官方 evaluator 集成测试。
- `results/`：审查快照中的旧结果，只作为可重新评估的历史方案来源。
- `results_round2/<experiment-id>/<run-id>/`：增量实验的隔离产物；默认由 CLI 创建，不覆盖旧 `results/`。

求解器和测试仅依赖 Python 标准库；图表使用标准库生成 SVG，不需要第三方绘图库。

## 单个用例

使用 Python 3.11 或更新版本，在工作区根目录运行，例如 Problem 1、4 核：

```bash
python3.11 -m solver.solver 2026_official/data/case_001.json \
  --problem 1 --ncores 4 --time-limit 300 --max-evals 8

单核基准默认单独最多运行 300 秒（`--baseline-timeout`），不挤占多核求解预算。若该图基准超时，错误标记会被后续核心数复用而不重复等待；确认要重试时显式加 `--retry-baseline`。
```

Problem 2 / Problem 3 分别将 `--problem` 改为 `2` / `3`。可一次传多个核心数，例如 `--ncores 2 3 4 5`。`--time-limit` 是每个 problem/core 组合的搜索与候选评估总预算秒数，`--max-evals` 限制多核候选 evaluator 调用数；单核固定基准另行计时和缓存。汇总区分基准、搜索、候选评估、求解器开销和总墙钟。方案先经过官方 `derive_multicore_plan` 校验；每次 evaluator 使用隔离临时目录及显式固定配置，不调用 shell 字符串。

程序打印的 `run_id` 用于定位结果目录和恢复批量实验。`--results-dir` 指向单个 run 叶目录；`--history-dir` 默认读取旧 `results/` 中的方案作为 warm-start 候选，但每个历史方案仍会在当前场景重新官方评估。官方评估成功后才更新 `best_validated`，未评估方案另存为 `legal_draft`。

旧入口仍可用：

```bash
python3.11 scheduler.py 2026_official/data/case_001.json -n 4 --problem 1
```

## 批量实验

先选代表性用例：

```bash
python3.11 experiments/run_all.py \
  --cases case_093 case_034 case_014 \
  --problems 1 2 3 --ncores 2 3 4 5 \
  --time-limit 300 --max-evals 1 --no-improve
```

扫描完整 100 个 case 时省略 `--cases`。命令会打印 `run_id`；中断后使用同一 ID 恢复，例如 `--resume --run-id <原run_id>`。恢复前会核对代码、配置、预算和计划矩阵，不匹配时不会静默跳过。结果默认写入 `results_round2/round2/<run_id>/`。

固定方案 Cache 2×2 对照和分区/分核消融：

```bash
python3.11 experiments/fixed_plan_cache_comparison.py --cases case_019 --ncores 4
python3.11 experiments/partition_schedule_ablation.py --case case_019 --ncores 4 --group-count 8
```

## 结果

- `schedules/`：每个 candidate 的方案、`legal_draft` 和仅由成功官方评估更新的 `best_validated`。
- `raw/`、`traces/`、`logs/`：按 run/candidate 唯一标记的官方结果、Trace 和日志；单核缓存附带输入与评估代码指纹。
- `candidate_trials.jsonl`、`subgraph_diagnostics.jsonl`：候选来源、官方结果、拒绝原因及逐子图逐核心代理估计。
- `summary.json`、`summary.csv`：按 experiment/run、代码/配置指纹、预算和 seed 隔离的结果行。
- `complete_case_manifest.json`、`summary_by_run.json`、`aggregate_summary.csv`：按 planned case manifest 统计成功、失败、未运行、单核参考缺失和配对覆盖。
- `figures/*.svg`：Problem 1/2 加速比、使用共同用例集的 Problem 3 L2 对照、Cache speedup、Makespan/新增搬运散点图。

重新汇总和绘图：

```bash
python3.11 experiments/collect_results.py --results-dir "results_round2/round2/<run_id>"
python3.11 experiments/make_figures.py --results-dir "results_round2/round2/<run_id>"
```

单核分母由官方 `singlecore_evaluate.py` 固定核内算法产生。Problem 3 的 `cache_hit_rate` 取官方 `cache_stats.hit_rate`，按命中字节占可缓存 COPY_IN 总字节计算。代理周期只用于列表分核，不能用作正式结果。

## 数据画像与测试

```bash
python3.11 experiments/profile_data.py
python3.11 -m unittest discover -s tests -v
```

`data_profile.csv` / `.md` 是 100 个官方 JSON 的校验与结构统计；其中 critical path 是 op DAG 周期估计，不是 evaluator Makespan。测试包含官方 6-cycle 最小图、依赖/切图环、场景 A/B 同核通信差异、跨核往返、全局 DDR 竞争、Cache 同时 miss、FIFO 不刷新及超容量不缓存。
