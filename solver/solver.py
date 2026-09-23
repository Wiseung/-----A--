from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

from . import cache_aware, schedule_a, schedule_b
from .evaluate_adapter import EvaluationError, Evaluator
from .graph_analysis import GraphAnalysis, analyze_graph
from .graph_io import Graph, load_config, load_graph
from .improve import generate_neighbors, plan_signature
from .legality import validate_plan


SUMMARY_FIELDS = [
    "case", "problem", "ncores", "method", "makespan",
    "single_core_baseline", "speedup", "added_copy_bytes",
    "original_copy_bytes", "scheduled_copy_bytes", "cache_hit_rate",
    "subgraph_count", "solver_runtime_sec", "evaluator_runtime_sec",
    "baseline_evaluator_runtime_sec", "evaluator_calls", "seed",
    "makespan_no_l2", "makespan_l2", "cache_speedup", "status", "error",
]


def _read_singlecore(results_dir: Path, graph: Graph) -> dict[str, Any] | None:
    path = results_dir / "raw" / f"{graph.path.stem}_singlecore.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _singlecore_failure(results_dir: Path, graph: Graph) -> Path:
    return results_dir / "logs" / f"{graph.path.stem}_singlecore.failed.txt"


def _save_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _store_summary(results_dir: Path, row: dict[str, Any]) -> None:
    json_path = results_dir / "summary.json"
    csv_path = results_dir / "summary.csv"
    previous = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else []
    key = (row["case"], row["problem"], row["ncores"])
    previous = [
        item for item in previous
        if (item.get("case"), item.get("problem"), item.get("ncores")) != key
    ]
    previous.append(row)
    previous.sort(key=lambda item: (item["case"], item["problem"], item["ncores"]))
    json_path.write_text(
        json.dumps(previous, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(previous)


def _objective(result: dict[str, Any]) -> tuple[int, int]:
    movement = result.get("data_movement_bytes", {})
    return result["makespan"], movement.get("added_copy_bytes", 0)


def _generator(problem: int) -> tuple[Callable[..., dict[str, Any]], Callable[..., list[int]], str]:
    if problem == 1:
        return schedule_a.generate_schedule, schedule_a.candidate_counts, "tensor_cut_critical_list"
    if problem == 2:
        return schedule_b.generate_schedule, schedule_b.candidate_counts, "co_location_critical_list"
    return cache_aware.generate_schedule, cache_aware.candidate_counts, "fifo_cache_aware_list"


def solve_one(
    graph: Graph,
    problem: int,
    ncores: int,
    config: dict[str, Any],
    evaluator: Evaluator,
    results_dir: Path,
    time_limit: float,
    max_evals: int,
    seed: int,
    improve: bool,
    baseline_timeout: float,
    retry_baseline: bool,
) -> dict[str, Any]:
    calls_before = evaluator.calls
    baseline_runtime = 0.0
    errors = []

    baseline = _read_singlecore(results_dir, graph)
    baseline_failure = _singlecore_failure(results_dir, graph)
    if baseline is None and baseline_failure.is_file() and not retry_baseline:
        errors.append("singlecore baseline previously failed; see " + str(baseline_failure))
    elif baseline is None:
        evaluator.timeout_sec = baseline_timeout
        try:
            single = evaluator.evaluate_singlecore(
                graph.path, f"{graph.path.stem}_singlecore"
            )
            baseline = single.result
            baseline_runtime = single.elapsed_sec
        except EvaluationError as error:
            baseline_runtime += error.elapsed_sec
            errors.append(f"singlecore baseline: {error}")

    solve_started = time.monotonic()
    deadline = solve_started + time_limit
    baseline_makespan = baseline.get("makespan") if baseline else None
    analysis = analyze_graph(graph)
    generate, get_counts, method = _generator(problem)
    group_counts = get_counts(analysis, ncores)
    best_plan: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    best_tag: str | None = None
    tested: set[str] = set()
    successful_evals = 0
    attempted_evals = 0
    multicore_eval_runtime = 0.0

    def try_candidate(plan: dict[str, Any], stage: str) -> bool:
        nonlocal best_plan, best_result, best_tag, successful_evals
        nonlocal attempted_evals, multicore_eval_runtime
        signature = plan_signature(plan)
        if signature in tested or attempted_evals >= max_evals:
            return False
        tested.add(signature)
        validate_plan(graph, plan, ncores)
        if best_plan is None:
            best_plan = plan
            _save_plan(
                results_dir / "schedules" /
                f"{graph.path.stem}_p{problem}_n{ncores}_best.json",
                best_plan,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        tag = (
            f"{graph.path.stem}_p{problem}_n{ncores}_"
            f"eval_{evaluator.calls + 1:04d}_{stage}"
        )
        evaluator.timeout_sec = max(0.1, remaining)
        attempted_evals += 1
        try:
            evaluated = evaluator.evaluate_multicore(
                graph.path, plan, problem, tag
            )
        except EvaluationError as error:
            multicore_eval_runtime += error.elapsed_sec
            errors.append(f"{tag}: {error}")
            return False
        multicore_eval_runtime += evaluated.elapsed_sec
        successful_evals += 1
        if best_result is None or _objective(evaluated.result) < _objective(best_result):
            best_plan = plan
            best_result = evaluated.result
            best_tag = tag
            _save_plan(
                results_dir / "schedules" /
                f"{graph.path.stem}_p{problem}_n{ncores}_best.json",
                best_plan,
            )
            return True
        return False

    for group_count in group_counts:
        if attempted_evals >= max_evals or time.monotonic() >= deadline:
            break
        try:
            plan = generate(analysis, ncores, config, group_count)
            try_candidate(plan, f"g{group_count}")
        except (ValueError, RuntimeError) as error:
            errors.append(f"initial groups={group_count}: {error}")

    if improve and best_result is not None and attempted_evals < max_evals:
        improved = True
        rounds = 0
        while improved and rounds < 2 and attempted_evals < max_evals:
            improved = False
            rounds += 1
            current = best_plan
            if current is None or time.monotonic() >= deadline:
                break
            for candidate in generate_neighbors(
                analysis, current, problem,
                limit=min(16, max_evals - attempted_evals),
            ):
                if attempted_evals >= max_evals or time.monotonic() >= deadline:
                    break
                improved = try_candidate(candidate, f"local{rounds}") or improved

    if best_plan is None:
        total_elapsed = time.monotonic() - solve_started
        row = {
            "case": graph.path.stem, "problem": problem, "ncores": ncores,
            "method": method, "makespan": None,
            "single_core_baseline": baseline_makespan, "speedup": None,
            "added_copy_bytes": None, "original_copy_bytes": None,
            "scheduled_copy_bytes": None, "cache_hit_rate": None,
            "subgraph_count": 0,
            "solver_runtime_sec": max(0.0, total_elapsed - multicore_eval_runtime),
            "evaluator_runtime_sec": baseline_runtime + multicore_eval_runtime,
            "baseline_evaluator_runtime_sec": baseline_runtime,
            "evaluator_calls": evaluator.calls - calls_before, "seed": seed,
            "makespan_no_l2": None, "makespan_l2": None,
            "cache_speedup": None, "status": "no_legal_candidate",
            "error": " | ".join(errors),
        }
        _store_summary(results_dir, row)
        return row

    movement = best_result.get("data_movement_bytes", {}) if best_result else {}
    makespan = best_result.get("makespan") if best_result else None
    speedup = (
        baseline_makespan / makespan
        if baseline_makespan is not None and makespan not in (None, 0)
        else None
    )
    l2_speedup = None
    makespan_no_l2 = None
    makespan_l2 = makespan if problem == 3 else None
    if problem == 3:
        summary_path = results_dir / "summary.json"
        if summary_path.is_file():
            for item in json.loads(summary_path.read_text(encoding="utf-8")):
                if item.get("case") == graph.path.stem and item.get("problem") == 2 and item.get("ncores") == ncores:
                    makespan_no_l2 = item.get("makespan")
                    break
        if makespan_no_l2 not in (None, 0) and makespan is not None:
            l2_speedup = makespan_no_l2 / makespan

    total_elapsed = time.monotonic() - solve_started
    evaluator_runtime = baseline_runtime + multicore_eval_runtime
    row = {
        "case": graph.path.stem,
        "problem": problem,
        "ncores": ncores,
        "method": method,
        "makespan": makespan,
        "single_core_baseline": baseline_makespan,
        "speedup": speedup,
        "added_copy_bytes": movement.get("added_copy_bytes"),
        "original_copy_bytes": movement.get("original_graph_copy_bytes"),
        "scheduled_copy_bytes": movement.get("scheduled_copy_bytes"),
        "cache_hit_rate": (best_result.get("cache_stats", {}).get("hit_rate")
                            if best_result else None),
        "subgraph_count": len(set(best_plan["node_to_subgraph"].values())),
        "solver_runtime_sec": max(0.0, total_elapsed - evaluator_runtime),
        "evaluator_runtime_sec": evaluator_runtime,
        "baseline_evaluator_runtime_sec": baseline_runtime,
        "evaluator_calls": evaluator.calls - calls_before,
        "seed": seed,
        "makespan_no_l2": makespan_no_l2,
        "makespan_l2": makespan_l2,
        "cache_speedup": l2_speedup,
        "status": "evaluated" if best_result else "legal_not_evaluated",
        "error": " | ".join(errors),
        "best_evaluation_tag": best_tag,
    }
    _store_summary(results_dir, row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate, validate, and evaluate a multi-core schedule.")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--problem", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("-n", "--ncores", "--num-cores", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--baseline-timeout", type=float, default=300.0)
    parser.add_argument("--retry-baseline", action="store_true")
    parser.add_argument("--max-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-improve", action="store_true")
    parser.add_argument("--official-root", type=Path, default=Path(__file__).resolve().parents[1] / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if (args.time_limit <= 0 or args.baseline_timeout <= 0
            or args.evaluator_timeout <= 0 or args.max_evals < 1):
        parser.error("time limits must be positive; max-evals must be at least 1")

    graph = load_graph(args.graph)
    config = load_config(args.official_root, args.config)
    evaluator = Evaluator(
        args.official_root,
        config["path"],
        args.results_dir,
        args.evaluator_timeout,
    )
    for ncores in args.ncores:
        if not 1 <= ncores <= 5:
            parser.error("ncores values must be between 1 and 5")
        row = solve_one(
            graph, args.problem, ncores, config, evaluator,
            args.results_dir.resolve(), args.time_limit, args.max_evals,
            args.seed, not args.no_improve, args.baseline_timeout,
            args.retry_baseline,
        )
        print(
            f"{row['case']} p{row['problem']} n{row['ncores']}: "
            f"status={row['status']} makespan={row['makespan']} "
            f"added_copy_bytes={row['added_copy_bytes']} "
            f"evals={row['evaluator_calls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())