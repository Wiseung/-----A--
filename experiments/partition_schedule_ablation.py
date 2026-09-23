from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.evaluate_adapter import EvaluationError, Evaluator
from solver.graph_analysis import analyze_graph
from solver.graph_io import load_config, load_graph
from solver.legality import validate_plan
from solver.partition import partition_contiguous
from solver.run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)
from solver.schedule_common import schedule_partition
from solver.solver import _plan_schedule_metrics


def _hash_plan(plan: dict[str, Any]) -> str:
    serialized = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate Problem 3 partition and placement heuristics on a fixed graph."
    )
    parser.add_argument("--case", default="case_019")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--group-count", type=int, default=8)
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if not 1 <= args.ncores <= 5 or args.group_count < 1 or args.evaluator_timeout <= 0:
        parser.error("ncores must be 1..5; group-count and evaluator-timeout must be positive")
    if not valid_run_label(args.case):
        parser.error("case must be a path-safe case stem")
    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")

    official_root = args.official_root.resolve()
    graph_path = official_root / "data" / f"{args.case}.json"
    if not graph_path.is_file():
        parser.error(f"unknown case: {args.case}")
    graph = load_graph(graph_path)
    analysis = analyze_graph(graph)
    config = load_config(official_root, args.config)
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else ROOT / "results_round2" / "r01_partition_schedule" / run_id
    )
    if results_dir.exists() and any(results_dir.iterdir()):
        parser.error("results directory is not empty; use a new run-id or directory")

    fingerprints = build_fingerprints(
        graph.path, config["path"], official_root, ROOT / "solver"
    )
    run_metadata = make_run_metadata(
        "r01_partition_schedule",
        run_id,
        fingerprints,
        {
            "case": args.case,
            "ncores": args.ncores,
            "requested_group_count": args.group_count,
            "evaluation_problem": 3,
            "evaluator_timeout_sec": args.evaluator_timeout,
        },
    )
    try:
        ensure_run_metadata(results_dir / "run_metadata.json", run_metadata)
    except ValueError as error:
        parser.error(str(error))

    evaluator = Evaluator(
        official_root, config["path"], results_dir, args.evaluator_timeout
    )
    comparisons = []
    diagnostics_by_candidate = []
    trial_lines = []
    for partition_problem in (2, 3):
        for schedule_problem in (2, 3):
            partition = partition_contiguous(
                analysis, args.group_count, problem=partition_problem
            )
            placement_diagnostics: dict[str, Any] = {}
            plan = schedule_partition(
                analysis,
                partition,
                args.ncores,
                problem=schedule_problem,
                config=config,
                diagnostics=placement_diagnostics,
            )
            validate_plan(graph, plan, args.ncores)
            plan_hash = _hash_plan(plan)
            candidate_id = (
                f"{args.case}_n{args.ncores}_partp{partition_problem}_"
                f"schedp{schedule_problem}_{run_id}"
            )
            plan_path = results_dir / "schedules" / f"{candidate_id}.json"
            _write_json(plan_path, plan)
            metrics = _plan_schedule_metrics(graph, plan)
            record: dict[str, Any] = {
                "experiment_id": "r01_partition_schedule",
                "run_id": run_id,
                "candidate_id": candidate_id,
                "case": args.case,
                "ncores": args.ncores,
                "partition_problem": partition_problem,
                "schedule_problem": schedule_problem,
                "evaluation_problem": 3,
                "requested_group_count": args.group_count,
                "actual_group_count": len(partition.groups),
                "plan_hash": plan_hash,
                "plan_path": _portable_path(plan_path),
                **fingerprints,
                **metrics,
            }
            try:
                evaluated = evaluator.evaluate_multicore(
                    graph.path,
                    plan,
                    3,
                    f"{candidate_id}_evalp3",
                )
                movement = evaluated.result.get("data_movement_bytes", {})
                record.update({
                    "status": "evaluated",
                    "official_makespan": evaluated.result.get("makespan"),
                    "added_copy_bytes": movement.get("added_copy_bytes"),
                    "partition_added_copy_bytes": movement.get(
                        "partition_added_copy_bytes"
                    ),
                    "spill_added_copy_bytes": movement.get("spill_added_copy_bytes"),
                    "cache_hit_rate": evaluated.result.get("cache_stats", {}).get(
                        "hit_rate"
                    ),
                    "evaluation_wall_time_sec": evaluated.elapsed_sec,
                })
            except EvaluationError as error:
                record.update({
                    "status": "evaluation_failed",
                    "error": str(error),
                    "evaluation_wall_time_sec": error.elapsed_sec,
                })
            comparisons.append(record)
            diagnostics_by_candidate.append({
                **record,
                "placement_diagnostics": placement_diagnostics,
            })
            trial_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))

    _write_json(results_dir / "partition_schedule_ablation.json", {
        "experiment_id": "r01_partition_schedule",
        "run_id": run_id,
        "case": args.case,
        "ncores": args.ncores,
        "requested_group_count": args.group_count,
        "comparisons": comparisons,
    })
    _write_json(results_dir / "subgraph_diagnostics.json", {
        "experiment_id": "r01_partition_schedule",
        "run_id": run_id,
        "candidates": diagnostics_by_candidate,
    })
    (results_dir / "candidate_trials.jsonl").write_text(
        "\n".join(trial_lines) + "\n",
        encoding="utf-8",
    )
    print(
        f"run_id={run_id} output={results_dir / 'partition_schedule_ablation.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
