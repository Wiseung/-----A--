from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.evaluate_adapter import EvaluationError, Evaluator
from solver.graph_io import load_config, load_graph
from solver.run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _objective(row: dict[str, Any]) -> tuple[float, float, float]:
    return tuple(
        float(row.get(field)) if row.get(field) is not None else float("inf")
        for field in ("makespan", "added_copy_bytes", "spill_added_copy_bytes")
    )


def _select_rows(
    rows: list[dict[str, Any]],
    top_k: int,
    arms: set[str],
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("evaluation_problem") == 2
        and row.get("status") in {"success", "equivalent_plan"}
        and row.get("arm") in arms
    ]
    by_arm_hash: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in eligible:
        key = (row["case"], row["ncores"], row["arm"], row["plan_hash"])
        current = by_arm_hash.get(key)
        if current is None or (_objective(row), row.get("group_count", 0)) < (
            _objective(current), current.get("group_count", 0)
        ):
            by_arm_hash[key] = row
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in by_arm_hash.values():
        grouped.setdefault(
            (row["case"], row["ncores"], row["arm"]), []
        ).append(row)
    selected_by_arm: list[dict[str, Any]] = []
    for group in grouped.values():
        selected_by_arm.extend(sorted(
            group,
            key=lambda row: (_objective(row), row.get("group_count", 0)),
        )[:top_k])
    selected_hashes = {
        (row["case"], row["ncores"], row["plan_hash"])
        for row in selected_by_arm
    }
    selected = [
        row for row in by_arm_hash.values()
        if (row["case"], row["ncores"], row["plan_hash"]) in selected_hashes
    ]
    return sorted(selected, key=lambda row: (
        row["case"], row["ncores"], row["plan_hash"], row["arm"],
        row.get("group_count", 0),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate selected P0 plans under official Problem 3."
    )
    parser.add_argument("--source-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=["contiguous_p2", "chain_contiguous"])
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument("--retain-traces", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.top_k < 1 or args.evaluator_timeout <= 0:
        parser.error("top-k and evaluator-timeout must be positive")
    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")
    source_dir = args.source_results_dir.resolve()
    source_path = source_dir / "phase4_matrix.json"
    if not source_path.is_file():
        parser.error(f"missing source phase4_matrix.json: {source_path}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory is not empty")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    selected = _select_rows(source.get("comparisons", []), args.top_k, set(args.arms))
    if not selected:
        parser.error("source matrix has no successful P2 rows for the requested arms")
    official_root = args.official_root.resolve()
    config = load_config(official_root, args.config)
    cases = sorted({row["case"] for row in selected})
    graphs = {
        case: load_graph(official_root / "data" / f"{case}.json")
        for case in cases
    }
    fingerprints = {
        case: build_fingerprints(
            graphs[case].path, config["path"], official_root, ROOT / "solver"
        )
        for case in cases
    }
    run_metadata = make_run_metadata(
        "r04_p0_selected_problem3",
        run_id,
        next(iter(fingerprints.values())),
        {
            "source_results_dir": _portable_path(source_dir),
            "source_matrix_hash": source.get("matrix_spec_hash"),
            "top_k": args.top_k,
            "selected_plan_count": len(selected),
            "evaluation_problem": 3,
            "retain_traces": args.retain_traces,
        },
    )
    ensure_run_metadata(output_dir / "run_metadata.json", run_metadata)
    evaluator = Evaluator(
        official_root,
        config["path"],
        output_dir,
        args.evaluator_timeout,
        retain_traces=args.retain_traces,
    )
    comparisons: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    ledger_path = output_dir / "candidate_trials.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")

    for index, source_row in enumerate(selected, start=1):
        plan_path = Path(source_row["plan_path"])
        if not plan_path.is_absolute():
            plan_path = ROOT / plan_path
        record = {
            key: source_row.get(key)
            for key in (
                "case", "ncores", "arm", "group_count", "plan_hash",
                "partition_strategy", "partition_problem", "placement_scoring",
                "cache_ordering", "makespan", "added_copy_bytes",
                "partition_added_copy_bytes", "spill_added_copy_bytes",
                "candidate_id", "source_results_dir", "source_run_id",
                "source_matrix_spec_hash",
            )
        }
        record.update({
            "problem_2_makespan": source_row.get("makespan"),
            "problem_2_added_copy_bytes": source_row.get("added_copy_bytes"),
            "problem_2_partition_added_copy_bytes": source_row.get(
                "partition_added_copy_bytes"
            ),
            "problem_2_spill_added_copy_bytes": source_row.get(
                "spill_added_copy_bytes"
            ),
        })
        record.update({
            "source_results_dir": source_row.get("source_results_dir")
            or _portable_path(source_dir),
            "source_plan_path": _portable_path(plan_path),
            "source_problem2_status": source_row.get("status"),
            "evaluation_problem": 3,
            "evaluation_source_candidate_id": source_row.get("candidate_id"),
            "evaluation_reused": False,
        })
        if not plan_path.is_file():
            record.update({"status": "invalid", "error": "source plan missing"})
        else:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            key = (record["case"], record["ncores"], record["plan_hash"])
            cached = cache.get(key)
            if cached is not None:
                cached_status = cached.get("status")
                record.update(cached)
                record.update({
                    "status": (
                        "equivalent_plan"
                        if cached_status in {"success", "equivalent_plan"}
                        else cached_status
                    ),
                    "evaluation_reused": True,
                })
            else:
                tag = f"{record['case']}_n{record['ncores']}_{record['arm']}_g{record['group_count']}_{run_id}_evalp3"
                try:
                    evaluated = evaluator.evaluate_multicore(
                        graphs[record["case"]].path, plan, 3, tag
                    )
                    result = evaluated.result
                    movement = result.get("data_movement_bytes", {})
                    cache_stats = result.get("cache_stats", {})
                    raw_path = output_dir / "raw" / f"{Evaluator._safe_tag(tag)}.json"
                    record.update({
                        "status": "success" if result.get("makespan") is not None else "invalid",
                        "makespan_p3": result.get("makespan"),
                        "problem_3_makespan": result.get("makespan"),
                        "added_copy_bytes_p3": movement.get("added_copy_bytes"),
                        "problem_3_added_copy_bytes": movement.get("added_copy_bytes"),
                        "partition_added_copy_bytes_p3": movement.get("partition_added_copy_bytes"),
                        "problem_3_partition_added_copy_bytes": movement.get(
                            "partition_added_copy_bytes"
                        ),
                        "spill_added_copy_bytes_p3": movement.get("spill_added_copy_bytes"),
                        "problem_3_spill_added_copy_bytes": movement.get(
                            "spill_added_copy_bytes"
                        ),
                        "cache_hit_bytes": cache_stats.get("hit_bytes"),
                        "cache_miss_bytes": cache_stats.get("miss_bytes"),
                        "cache_hit_rate": cache_stats.get("hit_rate"),
                        "raw_result_path": _portable_path(raw_path),
                        "raw_result_hash": _hash_file(raw_path) if raw_path.is_file() else None,
                        "evaluation_wall_time_sec": evaluated.elapsed_sec,
                        "error": None,
                    })
                except EvaluationError as error:
                    record.update({
                        "status": "timeout" if "timeout" in str(error).lower() else "evaluation_failed",
                        "error": str(error),
                        "evaluation_wall_time_sec": error.elapsed_sec,
                    })
                cache[key] = {
                    key: record.get(key)
                    for key in (
                        "status", "makespan_p3", "added_copy_bytes_p3",
                        "problem_3_makespan", "problem_3_added_copy_bytes",
                        "partition_added_copy_bytes_p3", "spill_added_copy_bytes_p3",
                        "problem_3_partition_added_copy_bytes",
                        "problem_3_spill_added_copy_bytes",
                        "cache_hit_bytes", "cache_miss_bytes", "cache_hit_rate",
                        "raw_result_path", "raw_result_hash", "evaluation_wall_time_sec",
                        "error",
                    )
                }
        comparisons.append(record)
        with ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[{index}/{len(selected)}] {record['case']} n{record['ncores']} {record['arm']} g{record['group_count']} {record['status']}", flush=True)

    _write_json(output_dir / "selected_problem3.json", {
        "experiment_id": "r04_p0_selected_problem3",
        "run_id": run_id,
        "source_matrix_hash": source.get("matrix_spec_hash"),
        "top_k": args.top_k,
        "comparisons": comparisons,
        "official_evaluation_calls": evaluator.calls,
    })
    _write_json(output_dir / "run_metadata.json", run_metadata | {
        "completed_count": len(comparisons),
        "official_evaluation_calls": evaluator.calls,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())