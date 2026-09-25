# 2026 NPU 多核调度求解器

本项目为 2026 华为杯 A 题多核 NPU 调度问题的可复现实验工程。求解器读取官方计算图，生成题目要求的 `node_to_subgraph` 和 `core_schedules`；最终 Makespan、数据搬运和 Cache 指标由 `2026_official/code/` 中的官方 evaluator 计算。

## 环境要求

- macOS、Linux 或其他能够运行官方 Python evaluator 的环境
- Python 3.11 或更高版本
- 项目运行和测试只依赖 Python 标准库
- 官方数据和 evaluator 必须保留在 `2026_official/`

本项目使用 `dataclass(slots=True)`，Python 3.9 环境不兼容。建议先确认：

```bash
python3.11 --version
```

所有命令均从仓库根目录执行。

## 快速验证

先运行完整回归测试：

```bash
python3.11 -m unittest discover -s tests -v
```

测试覆盖图解析、计划合法性、Problem 1/2/3 语义、官方 evaluator 集成、Cache 事件、候选账本和结果汇总。

运行单个用例，例如 Problem 1、4 核：

```bash
python3.11 -m solver.solver \
  2026_official/data/case_001.json \
  --problem 1 \
  --ncores 4 \
  --time-limit 300 \
  --max-evals 8
```

Problem 2 或 Problem 3 将 `--problem` 改为 `2` 或 `3`。多核候选的默认策略是 `ordered`，最终方案始终按官方 evaluator 的 `(makespan, added_copy_bytes)` 选择。

## 批量运行

运行指定的代表性用例：

```bash
python3.11 experiments/run_all.py \
  --cases case_093 case_034 case_014 \
  --problems 1 2 3 \
  --ncores 2 3 4 5 \
  --time-limit 300 \
  --max-evals 2 \
  --no-improve
```

扫描官方数据目录中的全部 `case_*.json` 时省略 `--cases`。每个批次都应使用独立的 `--results-dir` 或新的 `run_id`，不要覆盖已有正式结果。

中断后可以使用原 run id 恢复：

```bash
python3.11 experiments/run_all.py \
  --resume \
  --run-id <existing-run-id> \
  --results-dir results_round2/round2/<existing-run-id>
```

恢复操作会检查 case 矩阵、预算、代码指纹、配置和候选选择参数；身份不一致时会拒绝继续。

## 结果文件

每个 run 目录通常包含：

- `run_metadata.json`：experiment/run identity、代码哈希、配置哈希和预算
- `batch_manifest.json`：计划运行的 case、题目和核数矩阵
- `summary.json` / `summary.csv`：每个 case/problem/core 的最终结果
- `candidate_trials.jsonl`：候选来源、官方结果、筛选和拒绝原因
- `schedules/`：候选计划、`legal_draft` 和成功验证后的 `best_validated`
- `raw/`、`logs/`、`traces/`：官方 evaluator 输出和诊断
- `complete_case_manifest.json`：按 planned matrix 统计成功、失败和未运行组合
- `figures/`：SVG 图表和 `figure_manifest.json`

重新汇总和生成图表：

```bash
python3.11 experiments/collect_results.py \
  --results-dir results_round2/round2/<run-id>

python3.11 experiments/make_figures.py \
  --results-dir results_round2/round2/<run-id>
```

## 严格 P2/P3 Cache 配对

固定同一份 plan 分别运行官方 Problem 2 和 Problem 3 evaluator：

```bash
python3.11 experiments/fixed_plan_cache_comparison.py \
  --cases case_019 \
  --ncores 4 \
  --evaluator-timeout 120 \
  --results-dir results_round6/fixed_plan_pair_case019_n4
```

输出中的 `strict_pair_complete` 只有在以下条件同时满足时才为 `true`：

- P2 和 P3 都成功；
- 两条记录的 `plan_hash` 相同；
- 两条记录来自同一个 case/core 和同一份固定计划。

比较 FIFO 与 reuse-distance-inspired 排序：

```bash
python3.11 experiments/stratified_partition_cache_pair.py \
  --cases case_014 case_019 case_034 \
  --ncores 4 \
  --group-count 10 \
  --schedule-problem 2 \
  --cache-ordering fifo \
  --results-dir results_round6/p3_pair_fifo

python3.11 experiments/stratified_partition_cache_pair.py \
  --cases case_014 case_019 case_034 \
  --ncores 4 \
  --group-count 10 \
  --schedule-problem 2 \
  --cache-ordering reuse_distance \
  --results-dir results_round6/p3_pair_reuse
```

`reuse_distance` 当前是候选 ready-ordering 启发式，不是官方 Cache replacement policy。最终 Cache 结论必须使用官方 Problem 3 evaluator 的事件和指标。

## 实验模式

以下模式默认不启用，只用于独立消融：

```bash
# solver.py / run_all.py：候选代理 Pareto 筛选
--candidate-selection proxy_pareto

# solver.py / run_all.py：固定 core ownership 的 residence-aware 核内重排
--residence-ordering neighbor

# solver.py / run_all.py / stratified_partition_cache_pair.py：Cache 候选生成排序
--cache-ordering reuse_distance

# stratified_partition_cache_pair.py：共享 DDR 输入的首读者错峰候选
--shared-input-skew
```

这些模式不会自动替换正式保底方案。实验结果中必须区分：

- 官方 evaluator 的正式 Makespan；
- proxy、live-range 和 reuse-distance 诊断；
- 同一 plan hash 的 P2/P3 Cache 对照；
- 不同 plan 之间的算法候选比较。

## 官方结果口径

- 官方 evaluator 是最终性能结论的唯一依据。
- 代理周期、boundary copy、live-range、Cache priority 只能用于候选筛选和解释。
- Problem 3 的 Cache 效果不能通过不同 plan 的 P2/P3 结果直接相减推断。
- `screened_out` 和 `not_run_high_risk` 是候选预算状态，不是 evaluator 失败。
- 单核基线超时应单独记录，不能用任意多核结果替代单核分母。
- 不修改 `2026_official/code/`、官方数据或官方配置来配合实验。

## 项目说明

更详细的求解器结构、输出目录和旧入口说明见 [README_SOLVER.md](README_SOLVER.md)。最终正式结果和审计产物保存在本地 `results_round5/` 等隔离目录中；这些大体积实验产物不是运行本项目所必需的依赖。
