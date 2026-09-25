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


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the official single-core baseline for all cases."
    )
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument("--retain-traces", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.evaluator_timeout <= 0:
        parser.error("evaluator-timeout must be positive")
    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")
    official_root = args.official_root.resolve()
    config = load_config(official_root, args.config)
    graph_paths = sorted((official_root / "data").glob("case_*.json"))
    if not graph_paths:
        parser.error("official data directory has no case files")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        parser.error("output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {
        path.stem: build_fingerprints(
            path, config["path"], official_root, ROOT / "solver"
        )
        for path in graph_paths
    }
    run_metadata = make_run_metadata(
        "r05_singlecore_matrix",
        run_id,
        next(iter(fingerprints.values())),
        {
            "case_count": len(graph_paths),
            "evaluation_problem": "singlecore_baseline",
            "evaluator_timeout_sec": args.evaluator_timeout,
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
    ledger_path = output_dir / "candidate_trials.jsonl"
    if args.resume and ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                comparisons.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        ledger_path.write_text("", encoding="utf-8")
    completed_cases = {row.get("case") for row in comparisons}
    for index, graph_path in enumerate(graph_paths, start=1):
        case = graph_path.stem
        if case in completed_cases:
            continue
        tag = f"{case}_{run_id}_singlecore"
        record = {
            "case": case,
            "ncores": 1,
            "evaluation_problem": "singlecore_baseline",
            "evaluation_reused": False,
            "graph_hash": fingerprints[case]["graph_hash"],
            "config_hash": fingerprints[case]["config_hash"],
            "official_code_hash": fingerprints[case]["official_code_hash"],
            "solver_code_hash": fingerprints[case]["solver_code_hash"],
            "run_id": run_id,
        }
        try:
            evaluated = evaluator.evaluate_singlecore(graph_path, tag)
            result = evaluated.result
            raw_path = output_dir / "raw" / f"{Evaluator._safe_tag(tag)}.json"
            record.update({
                "status": "success" if result.get("makespan") is not None else "invalid",
                "makespan": result.get("makespan"),
                "raw_result_path": _portable_path(raw_path),
                "evaluation_wall_time_sec": evaluated.elapsed_sec,
                "error": None,
            })
        except EvaluationError as error:
            record.update({
                "status": "timeout" if "timeout" in str(error).lower()
                else "evaluation_failed",
                "evaluation_wall_time_sec": error.elapsed_sec,
                "error": str(error),
            })
        comparisons.append(record)
        with ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[{index}/{len(graph_paths)}] {case} singlecore {record['status']}",
            flush=True,
        )

    result = {
        "experiment_id": "r05_singlecore_matrix",
        "run_id": run_id,
        "evaluation_problem": "singlecore_baseline",
        "comparisons": comparisons,
        "official_evaluation_calls": len(comparisons),
        "status_counts": {
            status: sum(row.get("status") == status for row in comparisons)
            for status in sorted({row.get("status") for row in comparisons})
        },
    }
    _write_json(output_dir / "singlecore_matrix.json", result)
    _write_json(output_dir / "run_metadata.json", run_metadata | {
        "completed_count": len(comparisons),
        "official_evaluation_calls": len(comparisons),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())