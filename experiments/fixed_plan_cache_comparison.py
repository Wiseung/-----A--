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
from solver.graph_io import load_config, load_graph
from solver.legality import validate_plan
from solver.run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)
from solver.improve import plan_signature


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(plan_signature(plan).encode("utf-8")).hexdigest()


def _strict_pair_summary(
    source_problem: int,
    p2_record: dict[str, Any] | None,
    p3_record: dict[str, Any] | None,
) -> dict[str, Any]:
    p2_status = p2_record.get("status") if p2_record else "missing"
    p3_status = p3_record.get("status") if p3_record else "missing"
    p2_hash = p2_record.get("plan_hash") if p2_record else None
    p3_hash = p3_record.get("plan_hash") if p3_record else None
    strict_pair_complete = (
        p2_status in {"evaluated", "equivalent_plan"}
        and p3_status in {"evaluated", "equivalent_plan"}
        and p2_hash is not None
        and p2_hash == p3_hash
    )
    p2_makespan = p2_record.get("makespan") if p2_record else None
    p3_makespan = p3_record.get("makespan") if p3_record else None
    p3_over_p2 = (
        p3_makespan / p2_makespan
        if strict_pair_complete and p2_makespan not in (None, 0)
        and p3_makespan is not None
        else None
    )
    return {
        "source_problem": source_problem,
        "p2_status": p2_status,
        "p3_status": p3_status,
        "p2_plan_hash": p2_hash,
        "p3_plan_hash": p3_hash,
        "strict_pair_complete": strict_pair_complete,
        "p2_makespan": p2_makespan,
        "p3_makespan": p3_makespan,
        "p3_over_p2_makespan": p3_over_p2,
    }


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
        description="Cross-evaluate fixed Problem 2/3 plans under both official evaluators."
    )
    parser.add_argument("--cases", nargs="+", default=["case_019"])
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--source-results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.ncores < 1 or args.evaluator_timeout <= 0:
        parser.error("ncores and evaluator-timeout must be positive")
    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")

    official_root = args.official_root.resolve()
    data_dir = official_root / "data"
    available_cases = {path.stem: path for path in data_dir.glob("case_*.json")}
    missing = set(args.cases) - available_cases.keys()
    if missing:
        parser.error(f"unknown cases: {sorted(missing)}")
    config = load_config(official_root, args.config)
    source_results_dir = args.source_results_dir.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else ROOT / "results_round2" / "r00_fixed_plan_cache" / run_id
    )
    if results_dir.exists() and any(results_dir.iterdir()):
        parser.error("results directory is not empty; use a new run-id or directory")

    plan_sources = {}
    fingerprints_by_case = {}
    for case in args.cases:
        graph = load_graph(available_cases[case])
        fingerprints_by_case[case] = build_fingerprints(
            graph.path,
            config["path"],
            official_root,
            ROOT / "solver",
        )
        for source_problem in (2, 3):
            path = (
                source_results_dir / "schedules" /
                f"{case}_p{source_problem}_n{args.ncores}_best.json"
            )
            if not path.is_file():
                parser.error(f"missing source schedule: {path}")
            plan = json.loads(path.read_text(encoding="utf-8"))
            validate_plan(graph, plan, args.ncores)
            plan_sources[(case, source_problem)] = (path, plan)

    common_fingerprints = next(iter(fingerprints_by_case.values()))
    run_metadata = make_run_metadata(
        "r00_fixed_plan_cache",
        run_id,
        common_fingerprints,
        {
            "cases": sorted(args.cases),
            "ncores": args.ncores,
            "evaluator_timeout_sec": args.evaluator_timeout,
            "source_results_dir": str(source_results_dir),
            "source_plan_hashes": {
                f"{case}:p{source_problem}": _plan_hash(plan)
                for (case, source_problem), (_, plan) in plan_sources.items()
            },
        },
    )
    try:
        ensure_run_metadata(results_dir / "run_metadata.json", run_metadata)
    except ValueError as error:
        parser.error(str(error))

    evaluator = Evaluator(
        official_root,
        config["path"],
        results_dir,
        args.evaluator_timeout,
    )
    comparison = {
        "experiment_id": "r00_fixed_plan_cache",
        "run_id": run_id,
        "cases": [],
        "strict_pairs": [],
    }
    for case in args.cases:
        graph_path = available_cases[case]
        graph = load_graph(graph_path)
        evaluated: dict[tuple[int, int], dict[str, Any]] = {}
        records: dict[tuple[int, int], dict[str, Any]] = {}
        for source_problem in (2, 3):
            source_path, plan = plan_sources[(case, source_problem)]
            plan_hash = _plan_hash(plan)
            archived_plan = (
                results_dir / "fixed_plans" /
                f"{case}_n{args.ncores}_source_p{source_problem}_{plan_hash[:12]}.json"
            )
            _write_json(archived_plan, plan)
            for eval_problem in (2, 3):
                tag = (
                    f"{case}_n{args.ncores}_sourcep{source_problem}_"
                    f"evalp{eval_problem}_{run_id}"
                )
                record: dict[str, Any] = {
                    "case": case,
                    "ncores": args.ncores,
                    "source_problem": source_problem,
                    "evaluation_problem": eval_problem,
                    **fingerprints_by_case[case],
                    "plan_hash": plan_hash,
                    "source_plan": _portable_path(source_path),
                    "archived_plan": _portable_path(archived_plan),
                    "evaluation_tag": tag,
                }
                try:
                    result = evaluator.evaluate_multicore(
                        graph.path, plan, eval_problem, tag
                    )
                    movement = result.result.get("data_movement_bytes", {})
                    record.update({
                        "status": "evaluated",
                        "makespan": result.result.get("makespan"),
                        "added_copy_bytes": movement.get("added_copy_bytes"),
                        "partition_added_copy_bytes": movement.get(
                            "partition_added_copy_bytes"
                        ),
                        "spill_added_copy_bytes": movement.get("spill_added_copy_bytes"),
                        "cache_hit_rate": result.result.get("cache_stats", {}).get(
                            "hit_rate"
                        ),
                        "evaluation_wall_time_sec": result.elapsed_sec,
                    })
                    evaluated[(source_problem, eval_problem)] = record
                except EvaluationError as error:
                    record.update({
                        "status": "evaluation_failed",
                        "error": str(error),
                        "evaluation_wall_time_sec": error.elapsed_sec,
                    })
                records[(source_problem, eval_problem)] = record
                comparison["cases"].append(record)

            comparison["strict_pairs"].append({
                "case": case,
                "ncores": args.ncores,
                **_strict_pair_summary(
                    source_problem,
                    records.get((source_problem, 2)),
                    records.get((source_problem, 3)),
                ),
            })

        def ratio(numerator_key: tuple[int, int], denominator_key: tuple[int, int]) -> float | None:
            numerator = evaluated.get(numerator_key, {}).get("makespan")
            denominator = evaluated.get(denominator_key, {}).get("makespan")
            if numerator is None or denominator in (None, 0):
                return None
            return numerator / denominator

        comparison.setdefault("effects", {})[case] = {
            "hardware_effect_fixed_p2_plan": ratio((2, 2), (2, 3)),
            "cache_algorithm_adaptation": ratio((2, 3), (3, 3)),
            "combined_effect": ratio((2, 2), (3, 3)),
            "formula_convention": "makespan_numerator / makespan_denominator",
        }

    output_path = results_dir / "fixed_plan_cache_comparison.json"
    _write_json(output_path, comparison)
    print(f"run_id={run_id} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
