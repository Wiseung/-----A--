from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.evaluate_adapter import EvaluationError, Evaluator
from solver.features import graph_features
from solver.graph_analysis import analyze_graph
from solver.graph_io import load_config, load_graph
from solver.legality import validate_plan
from solver.partition import partition_with_strategy
from solver.run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)
from solver.schedule_common import schedule_partition
from solver.solver import _plan_schedule_metrics, plan_signature


EXPERIMENT_ID = "r03_stratified_partition_cache_pair"
DEFAULT_CASES = [
    "case_001", "case_014", "case_016", "case_019",
    "case_020", "case_028", "case_034", "case_093",
]
DEFAULT_GROUP_COUNTS = [8, 10, 12]
ARM_SPECS = {
    "contiguous_p2": {
        "partition_strategy": "contiguous",
        "partition_problem": 2,
        "placement_scoring": "baseline",
    },
    "contiguous_p3": {
        "partition_strategy": "contiguous",
        "partition_problem": 3,
        "placement_scoring": "baseline",
    },
    "cagg_lite": {
        "partition_strategy": "cagg_lite",
        "partition_problem": 2,
        "placement_scoring": "baseline",
    },
    "cagg_lite_coverage": {
        "partition_strategy": "cagg_lite_coverage",
        "partition_problem": 2,
        "placement_scoring": "soft",
    },
}
RESULT_FIELDS = (
    "makespan",
    "added_copy_bytes",
    "partition_added_copy_bytes",
    "spill_added_copy_bytes",
    "cache_hit_bytes",
    "cache_miss_bytes",
    "cache_hit_rate",
    "cache_hits",
    "cache_accesses",
    "memory_peak_by_core",
    "raw_result_path",
    "raw_result_hash",
    "evaluation_wall_time_sec",
    "error",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(plan_signature(plan).encode("utf-8")).hexdigest()


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

def _baseline_indexes() -> list[dict[str, str]]:
    paths = (
        ROOT / "results" / "summary.json",
        ROOT / "results_round2" / "r02_problem3_warm_start" / "r02-full-20260923"
        / "problem3_warm_start_ablation.csv",
        ROOT / "results_round3" / "r03_winner_reverse_p2" / "r03-reverse-p2-20260923"
        / "reverse_validation.csv",
    )
    return [
        {"path": _portable_path(path), "sha256": _hash_file(path)}
        for path in paths if path.is_file()
    ]


def _load_reuse_index(
    results_dir: Path | None,
    cases: list[str],
    ncores_values: list[int],
    group_count: int,
    partition_strategy: str,
    placement_scoring: str,
    fingerprints_by_case: dict[str, dict[str, str | None]],
) -> tuple[dict[tuple[str, int, int, int], dict[str, Any]], dict[str, str] | None]:
    if results_dir is None:
        return {}, None
    metadata_path = results_dir / "run_metadata.json"
    summary_path = results_dir / "partition_cache_pair.json"
    if not metadata_path.is_file() or not summary_path.is_file():
        raise ValueError("reuse results directory lacks run metadata or summary")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schedule_problem") != 2
        or metadata.get("requested_group_count") != group_count
        or metadata.get("partition_strategy", "contiguous") != partition_strategy
        or metadata.get("placement_scoring", "baseline") != placement_scoring
        or not set(cases).issubset(metadata.get("cases", []))
        or not set(ncores_values).issubset(metadata.get("ncores", []))
    ):
        raise ValueError("reuse results settings do not match the requested matrix")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {}
    for row in summary.get("comparisons", []):
        case = row.get("case")
        if case not in fingerprints_by_case or row.get("schedule_problem") != 2:
            continue
        if any(
            row.get(field) != fingerprints_by_case[case].get(field)
            for field in (
                "graph_hash", "config_hash", "official_code_hash", "solver_code_hash"
            )
        ):
            continue
        if row.get("status") not in {"success", "equivalent_plan"}:
            continue
        key = (
            case,
            row["ncores"],
            row["partition_problem"],
            row["evaluation_problem"],
        )
        rows[key] = row
    identity = {
        "summary_path": _portable_path(summary_path),
        "summary_sha256": _hash_file(summary_path),
        "run_id": metadata.get("run_id", ""),
    }
    return rows, identity


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["status"] for row in rows)
    return dict(sorted(counts.items()))


def _candidate_matrix(
    cases: list[str],
    ncores_values: list[int],
    group_counts: list[int],
    arms: list[str],
    run_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"{case}_n{ncores}_{arm}_g{group_count}_{run_id}",
            "case": case,
            "ncores": ncores,
            "arm": arm,
            "group_count": group_count,
            **ARM_SPECS[arm],
        }
        for case in cases
        for ncores in ncores_values
        for arm in arms
        for group_count in group_counts
    ]


def _proxy_score(
    plan_metrics: dict[str, Any],
    placement_diagnostics: dict[str, Any],
) -> tuple[float, float, float, int, int, int]:
    estimated_makespan = max(
        (group.get("group_finish", 0.0)
         for group in placement_diagnostics.get("groups", [])),
        default=0.0,
    )
    added_copy_bytes = placement_diagnostics.get(
        "estimated_partition_added_copy_bytes_recomputed"
    )
    if added_copy_bytes is None:
        added_copy_bytes = placement_diagnostics.get(
            "estimated_partition_added_copy_bytes"
        )
    if added_copy_bytes is None:
        added_copy_bytes = 10**30
    repeated_input_bytes = placement_diagnostics.get(
        "estimated_repeated_input_bytes", 0
    )
    imbalance = max(
        plan_metrics.get("m_load_imbalance", 0.0),
        plan_metrics.get("v_load_imbalance", 0.0),
    )
    return (
        float(estimated_makespan),
        float(plan_metrics.get("max_core_compute", 0)),
        float(imbalance),
        -int(plan_metrics.get("active_core_count", 0)),
        int(added_copy_bytes),
        int(repeated_input_bytes),
    )


def _select_proxy_candidates(
    candidates: list[dict[str, Any]],
    proxy_screen: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (candidate["case"], candidate["ncores"], candidate["arm"])
        grouped.setdefault(key, []).append(candidate)
    selected_ids: set[str] = set()
    for group in grouped.values():
        valid = sorted(
            (candidate for candidate in group if candidate.get("proxy_score") is not None),
            key=lambda candidate: (
                tuple(candidate["proxy_score"]),
                candidate["group_count"],
                candidate["candidate_id"],
            ),
        )
        for rank, candidate in enumerate(valid, start=1):
            candidate["proxy_rank"] = rank
        chosen = valid[:1] if proxy_screen else valid
        for candidate in chosen:
            selected_ids.add(candidate["candidate_id"])
    for candidate in candidates:
        selected = candidate["candidate_id"] in selected_ids
        candidate["proxy_selected"] = selected
        if candidate.get("proxy_score") is None:
            candidate["candidate_status"] = "invalid"
        else:
            candidate["candidate_status"] = (
                "proxy_selected" if selected else "screened_out"
            )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare P2/P3 partitions under paired Problem 2/3 evaluation."
        )
    )
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--ncores", nargs="+", type=int, default=[2, 4, 5])
    parser.add_argument("--group-count", type=int, default=10)
    parser.add_argument(
        "--partition-strategy", choices=(
            "contiguous", "cagg_lite", "cagg_lite_coverage"
        ),
        default="contiguous",
    )
    parser.add_argument(
        "--placement-scoring", choices=("baseline", "soft"),
        default="baseline",
    )
    parser.add_argument("--schedule-problem", type=int, choices=(2, 3), default=2)
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--reuse-results-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument("--retain-traces", action="store_true")
    args = parser.parse_args()

    if not args.cases or len(set(args.cases)) != len(args.cases):
        parser.error("cases must be non-empty and unique")
    if any(not valid_run_label(case) for case in args.cases):
        parser.error("case names must be path-safe labels")
    if (not args.ncores or len(set(args.ncores)) != len(args.ncores)
            or any(not 1 <= ncores <= 5 for ncores in args.ncores)):
        parser.error("ncores must be unique values in 1..5")
    if args.group_count < 1 or args.evaluator_timeout <= 0:
        parser.error("group-count and evaluator-timeout must be positive")

    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")
    official_root = args.official_root.resolve()
    config = load_config(official_root, args.config)
    graph_paths = {
        case: official_root / "data" / f"{case}.json" for case in args.cases
    }
    missing = [case for case, path in graph_paths.items() if not path.is_file()]
    if missing:
        parser.error(f"unknown cases: {missing}")

    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else ROOT / "results_round3" / EXPERIMENT_ID / run_id
    )
    if results_dir.exists() and any(results_dir.iterdir()):
        parser.error("results directory is not empty; use a new run-id or directory")

    fingerprints_by_case = {
        case: build_fingerprints(
            graph_path, config["path"], official_root, ROOT / "solver"
        )
        for case, graph_path in graph_paths.items()
    }
    common_fingerprints = next(iter(fingerprints_by_case.values()))
    baseline_indexes = _baseline_indexes()
    try:
        reuse_rows, reuse_identity = _load_reuse_index(
            args.reuse_results_dir.resolve() if args.reuse_results_dir else None,
            args.cases,
            args.ncores,
            args.group_count,
            args.partition_strategy,
            args.placement_scoring,
            fingerprints_by_case,
        )
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        parser.error(str(error))
    run_metadata = make_run_metadata(
        EXPERIMENT_ID,
        run_id,
        common_fingerprints,
        {
            "cases": args.cases,
            "ncores": args.ncores,
            "requested_group_count": args.group_count,
            "partition_strategy": args.partition_strategy,
            "placement_scoring": args.placement_scoring,
            "partition_problems": [2, 3],
            "schedule_problem": args.schedule_problem,
            "evaluation_problems": [2, 3],
            "evaluator_timeout_sec": args.evaluator_timeout,
            "retain_traces": args.retain_traces,
            "official_trace_output_generated": True,
            "seed": None,
            "experiment_script_hash": _hash_file(Path(__file__).resolve()),
            "baseline_indexes": baseline_indexes,
            "reuse_results_index": reuse_identity,
        },
    )
    try:
        ensure_run_metadata(results_dir / "run_metadata.json", run_metadata)
    except ValueError as error:
        parser.error(str(error))

    planned = [
        {
            "case": case,
            "ncores": ncores,
            "partition_problem": partition_problem,
            "schedule_problem": args.schedule_problem,
            "evaluation_problem": evaluation_problem,
            "placement_scoring": args.placement_scoring,
            "status": "not_run",
        }
        for case in args.cases
        for ncores in args.ncores
        for partition_problem in (2, 3)
        for evaluation_problem in (2, 3)
    ]
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "cases": args.cases,
        "ncores": args.ncores,
        "requested_group_count": args.group_count,
        "partition_strategy": args.partition_strategy,
        "placement_scoring": args.placement_scoring,
        "partition_problems": [2, 3],
        "schedule_problem": args.schedule_problem,
        "evaluation_problems": [2, 3],
        "planned_candidate_evaluations": len(planned),
        "baseline_indexes": baseline_indexes,
        "planned": planned,
    }
    _write_json(results_dir / "batch_manifest.json", manifest)

    features_by_case = {}
    graphs = {}
    analyses = {}
    for case, graph_path in graph_paths.items():
        graph = load_graph(graph_path)
        analysis = analyze_graph(graph)
        features = graph_features(analysis)
        features["max_tensor_fanout"] = max(
            (
                len(consumers & analysis.eligible_ops)
                for consumers in graph.tensor_consumers.values()
            ),
            default=0,
        )
        features_by_case[case] = features
        graphs[case] = graph
        analyses[case] = analysis
    _write_json(results_dir / "graph_features.json", {
        "fingerprints_by_case": fingerprints_by_case,
        "features_by_case": features_by_case,
    })

    evaluator = Evaluator(
        official_root,
        config["path"],
        results_dir,
        args.evaluator_timeout,
        retain_traces=args.retain_traces,
    )
    comparisons: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    evaluation_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    ledger_path = results_dir / "candidate_trials.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")

    def persist() -> None:
        outcomes = {
            (
                row["case"], row["ncores"], row["partition_problem"],
                row["evaluation_problem"],
            ): row
            for row in comparisons
        }
        for item in planned:
            key = (
                item["case"], item["ncores"], item["partition_problem"],
                item["evaluation_problem"],
            )
            outcome = outcomes.get(key)
            if outcome is not None:
                for field in (
                    "status", "candidate_id", "plan_hash", "makespan",
                    "evaluation_reused", "evaluation_source_candidate_id",
                    "raw_result_path", "error",
                ):
                    if field in outcome:
                        item[field] = outcome[field]
        manifest["status"] = "running"
        manifest["completed_candidate_evaluations"] = len(comparisons)
        manifest["unique_official_evaluations"] = evaluator.calls
        manifest["status_counts"] = _status_counts(comparisons)
        _write_json(results_dir / "batch_manifest.json", manifest)
        _write_json(results_dir / "partition_cache_pair.json", {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "requested_group_count": args.group_count,
            "partition_strategy": args.partition_strategy,
            "placement_scoring": args.placement_scoring,
            "schedule_problem": args.schedule_problem,
            "evaluation_problems": [2, 3],
            "status_counts": _status_counts(comparisons),
            "comparisons": comparisons,
        })

    persist()
    total = len(planned)
    done = 0
    print(f"run_id={run_id} results_dir={results_dir}", flush=True)

    for case in args.cases:
        graph = graphs[case]
        analysis = analyses[case]
        fingerprints = fingerprints_by_case[case]
        for ncores in args.ncores:
            for partition_problem in (2, 3):
                candidate_id = (
                    f"{case}_n{ncores}_partp{partition_problem}_"
                    f"schedp{args.schedule_problem}_{run_id}"
                )
                generation_started = time.monotonic()
                placement_diagnostics: dict[str, Any] = {}
                try:
                    partition = partition_with_strategy(
                        analysis,
                        args.group_count,
                        problem=partition_problem,
                        partition_strategy=args.partition_strategy,
                    )
                    plan = schedule_partition(
                        analysis,
                        partition,
                        ncores,
                        problem=args.schedule_problem,
                        config=config,
                        diagnostics=placement_diagnostics,
                        placement_scoring=args.placement_scoring,
                    )
                    validate_plan(graph, plan, ncores)
                    plan_hash = _plan_hash(plan)
                    plan_metrics = _plan_schedule_metrics(graph, plan, analysis)
                    generation_time = time.monotonic() - generation_started
                    plan_path = results_dir / "schedules" / f"{candidate_id}.json"
                    _write_json(plan_path, plan)
                    base_record = {
                        "experiment_id": EXPERIMENT_ID,
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                        "case": case,
                        "ncores": ncores,
                        "partition_problem": partition_problem,
                        "schedule_problem": args.schedule_problem,
                        "requested_group_count": args.group_count,
                        "partition_strategy": args.partition_strategy,
                        "placement_scoring": args.placement_scoring,
                        "actual_group_count": len(partition.groups),
                        "plan_hash": plan_hash,
                        "plan_path": _portable_path(plan_path),
                        "plan_generation_wall_time_sec": generation_time,
                        "graph_features_path": _portable_path(
                            results_dir / "graph_features.json"
                        ),
                        **fingerprints,
                        **plan_metrics,
                    }
                    diagnostics.append({
                        **base_record,
                        "placement_diagnostics": placement_diagnostics,
                    })
                    _write_json(results_dir / "subgraph_diagnostics.json", {
                        "candidates": diagnostics,
                    })
                    generation_error = None
                except (ValueError, KeyError, TypeError) as error:
                    generation_error = str(error)
                    base_record = {
                        "experiment_id": EXPERIMENT_ID,
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                        "case": case,
                        "ncores": ncores,
                        "partition_problem": partition_problem,
                        "schedule_problem": args.schedule_problem,
                        "requested_group_count": args.group_count,
                        "partition_strategy": args.partition_strategy,
                        "placement_scoring": args.placement_scoring,
                        "plan_hash": None,
                        "plan_path": None,
                        "plan_generation_wall_time_sec": time.monotonic() - generation_started,
                        "error": generation_error,
                        **fingerprints,
                    }

                for evaluation_problem in (2, 3):
                    done += 1
                    record = {
                        **base_record,
                        "evaluation_problem": evaluation_problem,
                        "evaluation_reused": False,
                    }
                    if generation_error is not None:
                        record.update({"status": "invalid", "error": generation_error})
                    else:
                        cache_key = (case, ncores, evaluation_problem, plan_hash)
                        cached = evaluation_cache.get(cache_key)
                        if cached is None:
                            source = reuse_rows.get((
                                case, ncores, partition_problem, evaluation_problem
                            ))
                            if source is not None and source.get("plan_hash") == plan_hash:
                                source_path = ROOT / source.get("raw_result_path", "")
                                source_hash = source.get("raw_result_hash")
                                if (
                                    source_path.is_file()
                                    and source_hash
                                    and _hash_file(source_path) == source_hash
                                ):
                                    source_result = json.loads(
                                        source_path.read_text(encoding="utf-8")
                                    )
                                    if source_result.get("makespan") == source.get("makespan"):
                                        cached = {
                                            key: source.get(key) for key in RESULT_FIELDS
                                        }
                                        cached.update({
                                            "status": "success",
                                            "evaluation_source_candidate_id": source.get(
                                                "evaluation_source_candidate_id",
                                                source["candidate_id"],
                                            ),
                                            "evaluation_source_run_id": source["run_id"],
                                        })
                                        evaluation_cache[cache_key] = cached
                        if cached is not None:
                            record.update(cached)
                            record.update({
                                "status": "equivalent_plan",
                                "evaluation_source_status": cached["status"],
                                "evaluation_reused": True,
                                "evaluation_source_candidate_id": cached[
                                    "evaluation_source_candidate_id"
                                ],
                            })
                        else:
                            tag = f"{candidate_id}_evalp{evaluation_problem}"
                            try:
                                evaluated = evaluator.evaluate_multicore(
                                    graph.path, plan, evaluation_problem, tag
                                )
                                result = evaluated.result
                                movement = result.get("data_movement_bytes", {})
                                cache_stats = result.get("cache_stats", {})
                                raw_path = (
                                    results_dir / "raw" /
                                    f"{Evaluator._safe_tag(tag)}.json"
                                )
                                makespan = result.get("makespan")
                                record.update({
                                    "status": "success" if makespan is not None else "invalid",
                                    "makespan": makespan,
                                    "added_copy_bytes": movement.get("added_copy_bytes"),
                                    "partition_added_copy_bytes": movement.get(
                                        "partition_added_copy_bytes"
                                    ),
                                    "spill_added_copy_bytes": movement.get(
                                        "spill_added_copy_bytes"
                                    ),
                                    "cache_hit_bytes": cache_stats.get("hit_bytes"),
                                    "cache_miss_bytes": cache_stats.get("miss_bytes"),
                                    "cache_hit_rate": cache_stats.get("hit_rate"),
                                    "cache_hits": cache_stats.get("hits"),
                                    "cache_accesses": cache_stats.get("accesses"),
                                    "memory_peak_by_core": result.get(
                                        "memory_peak_by_core"
                                    ),
                                    "raw_result_path": _portable_path(raw_path),
                                    "raw_result_hash": (
                                        _hash_file(raw_path) if raw_path.is_file() else None
                                    ),
                                    "evaluation_wall_time_sec": evaluated.elapsed_sec,
                                    "error": None,
                                    "evaluation_source_candidate_id": candidate_id,
                                })
                            except EvaluationError as error:
                                status = (
                                    "timeout" if "timeout" in str(error).lower()
                                    else "evaluation_failed"
                                )
                                record.update({
                                    "status": status,
                                    "evaluation_wall_time_sec": error.elapsed_sec,
                                    "error": str(error),
                                    "evaluation_source_candidate_id": candidate_id,
                                })
                            evaluation_cache[cache_key] = {
                                key: record.get(key) for key in RESULT_FIELDS
                            }
                            evaluation_cache[cache_key]["status"] = record["status"]
                            evaluation_cache[cache_key][
                                "evaluation_source_candidate_id"
                            ] = record.get("evaluation_source_candidate_id", candidate_id)
                            evaluation_cache[cache_key]["evaluation_source_run_id"] = run_id

                    comparisons.append(record)
                    with ledger_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        stream.flush()
                    persist()
                    print(
                        f"[{done}/{total}] {case} n{ncores} partP{partition_problem} "
                        f"evalP{evaluation_problem} {record['status']} "
                        f"makespan={record.get('makespan')}",
                        flush=True,
                    )

    manifest["status"] = "completed"
    manifest["completed_candidate_evaluations"] = len(comparisons)
    manifest["unique_official_evaluations"] = evaluator.calls
    manifest["status_counts"] = _status_counts(comparisons)
    _write_json(results_dir / "batch_manifest.json", manifest)
    _write_json(results_dir / "run_metadata.json", run_metadata | {
        "unique_official_evaluations": evaluator.calls,
        "completed_candidate_evaluations": len(comparisons),
        "status_counts": _status_counts(comparisons),
    })
    print(
        f"completed={len(comparisons)} unique_official_evaluations={evaluator.calls} "
        f"summary={results_dir / 'partition_cache_pair.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())