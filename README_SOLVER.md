# 2026 多核调度求解器

此工程读取 2026 官方 `tensors/ops/edges` 计算图，只输出赛题规定的 `node_to_subgraph` 和 `core_schedules`。实际 Makespan、数据搬运与 Cache 指标均由 `2026_official/code/` 官方评估器计算。

## 目录

- `2026_official/`：原附件 ZIP 的原样解压目录；ZIP 保留在工作区根目录。
- `solver/`：图解析、依赖分析、特征、连续拓扑分块、合法性、场景调度、评估适配器和有界局部邻域。
- `experiments/`：正式数据画像、批量运行、结果汇总和 SVG 图生成。
- `tests/`：标准库 unittest 微型图和官方 evaluator 集成测试。
- `results/`：方案、结果 JSON、Trace、日志、汇总表和图表。

求解器和测试仅依赖 Python 标准库；图表使用标准库生成 SVG，不需要第三方绘图库。

## 单个用例

在工作区根目录运行，例如 Problem 1、4 核：

```bash
python -m solver.solver 2026_official/data/case_001.json \
  --problem 1 --ncores 4 --time-limit 300 --max-evals 8

单核基准默认单独最多运行 300 秒（`--baseline-timeout`），不挤占多核求解预算。若该图基准超时，错误标记会被后续核心数复用而不重复等待；确认要重试时显式加 `--retry-baseline`。
```

Problem 2 / Problem 3 分别将 `--problem` 改为 `2` / `3`。可一次传多个核心数，例如 `--ncores 2 3 4 5`。`--time-limit` 是每个 problem/core 组合的求解与评估总预算秒数，`--max-evals` 限制多核候选 evaluator 调用数；单核固定基准另调用一次并保存。`--no-improve` 关闭局部邻域。方案先经过官方 `derive_multicore_plan` 校验；每次 evaluator 使用隔离临时目录及显式固定配置，不调用 shell 字符串。

旧入口仍可用：

```bash
python scheduler.py 2026_official/data/case_001.json -n 4 --problem 1
```

## 批量实验

先选代表性用例：

```bash
python experiments/run_all.py \
  --cases case_093 case_034 case_014 \
  --problems 1 2 3 --ncores 2 3 4 5 \
  --time-limit 300 --max-evals 1 --no-improve
```

扫描完整 100 个 case 时省略 `--cases`。中断后可用 `--resume` 跳过已标记为 evaluated 的组合。完整 100×3×5 组合耗时较长，先以代表性图验证，再按剩余预算扩展。

## 结果

- `results/schedules/`：每个组合当前最佳合法方案，候选评估前先落盘。
- `results/raw/`：官方 evaluator 结果 JSON 和单核基准结果。
- `results/traces/`、`results/logs/`：官方 Perfetto Trace 与简短日志。
- `results/summary.json`、`results/summary.csv`：每组合 makespan、固定单核基准、加速比、搬运量、Cache 字节命中率、运行时间及 evaluator 调用数。
- `results/aggregate_summary.csv`：按逐用例加速比算术平均生成的曲线数据；缺失组合单独计数。
- `results/figures/*.svg`：Problem 1/2 加速比、Problem 3 L2 对照、Cache speedup、Makespan/新增搬运散点图。SVG 中 `k` 为实际纳入计算的用例数。

重新汇总和绘图：

```bash
python experiments/collect_results.py
python experiments/make_figures.py
```

单核分母由官方 `singlecore_evaluate.py` 固定核内算法产生。Problem 3 的 `cache_hit_rate` 取官方 `cache_stats.hit_rate`，按命中字节占可缓存 COPY_IN 总字节计算。代理周期只用于列表分核，不能用作正式结果。

## 数据画像与测试

```bash
python experiments/profile_data.py
python -m unittest discover -s tests -v
```

`data_profile.csv` / `.md` 是 100 个官方 JSON 的校验与结构统计；其中 critical path 是 op DAG 周期估计，不是 evaluator Makespan。测试包含官方 6-cycle 最小图、依赖/切图环、场景 A/B 同核通信差异、跨核往返、全局 DDR 竞争、Cache 同时 miss、FIFO 不刷新及超容量不缓存。
