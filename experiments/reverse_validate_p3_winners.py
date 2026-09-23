from __future__ import annotations

import argparse
import csv
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
from solver.solver import plan_signature


EXPERIMENT_ID = "r03_winner_reverse_p2"
CSV_FIELDS = [
    "case",
    "ncores",
    "candidate_id",
    "candidate_family",
    "source_plan_path",
    "plan_path",
    "plan_hash",
    "source_p3_result_path",
    "source_p3_result_hash",
    "source_p3_makespan",
    "p2_baseline_makespan",
    "p2_makespan",
    "p2_delta_vs_baseline",
    "p2_relative_change_vs_baseline",
    "p3_warm_makespan",
    "p3_mixed_makespan",
    "p2_status",
    "p2_raw_result_path",
    "p2_raw_result_hash",
    "evaluation_wall_time_sec",
    "error",
]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(plan_signature(plan).encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_winner_rows(source_dir: Path) -> list[dict[str, Any]]:
    csv_path = source_dir / "problem3_warm_start_ablation.csv"
    trial_path = source_dir / "candidate_trials.jsonl"
    if not csv_path.is_file() or not trial_path.is_file():
        raise ValueError("source run must include the R02 CSV and candidate_trials.jsonl")

    trials = {}
    for line_number, line in enumerate(trial_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            trial = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid candidate trial at line {line_number}") from error
        if trial.get("candidate_id"):
            trials[trial["candidate_id"]] = trial

    winners = []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            warm = row.get("p2_to_p3_warm_makespan")
            mixed = row.get("p2_to_p3_mixed_makespan")
            candidate_id = row.get("p2_to_p3_mixed_candidate_id")
            if not (warm and mixed and candidate_id):
                continue
            if int(mixed) >= int(warm):
                continue
            trial = trials.get(candidate_id)
            if trial is None or trial.get("status") != "evaluated":
                raise ValueError(f"winner {candidate_id} has no successful candidate trial")
            winners.append({**row, "trial": trial})
    if not winners:
        raise ValueError("source run contains no strictly improved mixed winners")
    return winners


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate strict Problem 3 mixed winners with the Problem 2 evaluator."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "results_round2/r02_problem3_warm_start/r02-full-20260923",
    )
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--run-id", default="r03-reverse-p2-20260923")
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument("--retain-traces", action="store_true")
    args = parser.parse_args()

    if not valid_run_label(args.run_id) or args.evaluator_timeout <= 0:
        parser.error("run-id must be path-safe and evaluator-timeout must be positive")
    source_dir = args.source_dir.resolve()
    official_root = args.official_root.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else ROOT / "results_round3" / EXPERIMENT_ID / args.run_id
    )
    if results_dir.exists() and any(results_dir.iterdir()):
        parser.error("results directory is not empty; use a new run-id or directory")

    source_metadata = json.loads(
        (source_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    config = load_config(official_root, args.config)
    winners = _load_winner_rows(source_dir)
    cases = sorted({row["case"] for row in winners})
    graph_paths = {
        case: official_root / "data" / f"{case}.json"
        for case in cases
    }
    fingerprints_by_case = {
        case: build_fingerprints(
            graph_paths[case], config["path"], official_root, ROOT / "solver"
        )
        for case in cases
    }
    current_config_hash = next(iter(fingerprints_by_case.values()))["config_hash"]
    current_official_hash = next(iter(fingerprints_by_case.values()))["official_code_hash"]
    if source_metadata.get("config_hash") != current_config_hash:
        parser.error("source config hash differs from the current evaluation config")
    if source_metadata.get("official_code_hash") != current_official_hash:
        parser.error("source official code hash differs from the current evaluator")
    for case in cases:
        if source_metadata.get("graph_hashes", {}).get(case) != fingerprints_by_case[case]["graph_hash"]:
            parser.error(f"source graph hash differs for {case}")

    run_metadata = make_run_metadata(
        EXPERIMENT_ID,
        args.run_id,
        next(iter(fingerprints_by_case.values())),
        {
            "source_experiment_id": source_metadata.get("experiment_id"),
            "source_run_id": source_metadata.get("run_id"),
            "source_dir": _portable_path(source_dir),
            "evaluation_problem": 2,
            "evaluator_timeout_sec": args.evaluator_timeout,
            "retain_traces": args.retain_traces,
            "official_trace_output_generated": True,
            "candidate_count": len(winners),
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
        retain_traces=args.retain_traces,
    )
    records = []
    trial_lines = []
    for row in winners:
        case = row["case"]
        ncores = int(row["ncores"])
        candidate_id = row["p2_to_p3_mixed_candidate_id"]
        trial = row["trial"]
        source_plan = source_dir / "schedules" / f"{candidate_id}.json"
        source_p3_result = source_dir / "raw" / f"{candidate_id}.json"
        if not source_plan.is_file() or not source_p3_result.is_file():
            parser.error(f"winner {candidate_id} is missing its plan or P3 result")
        try:
            plan = json.loads(source_plan.read_text(encoding="utf-8"))
            p3_result = json.loads(source_p3_result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"winner {candidate_id} source artifact is invalid: {error}")
        graph = load_graph(graph_paths[case])
        validate_plan(graph, plan, ncores)
        plan_hash = _plan_hash(plan)
        if plan_hash != row["p2_to_p3_mixed_plan_hash"]:
            parser.error(f"winner {candidate_id} plan hash differs from the R02 CSV")
        if p3_result.get("makespan") != int(row["p2_to_p3_mixed_makespan"]):
            parser.error(f"winner {candidate_id} P3 result differs from the R02 CSV")
        if trial.get("plan_hash") != plan_hash or trial.get("accepted") is not True:
            parser.error(f"winner {candidate_id} candidate ledger does not match its plan")

        archived_plan = results_dir / "schedules" / f"{candidate_id}_p2_reverse.json"
        _write_json(archived_plan, plan)
        tag = f"{candidate_id}_reverse_p2"
        record: dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": args.run_id,
            "case": case,
            "ncores": ncores,
            "problem": 2,
            "candidate_id": candidate_id,
            "candidate_family": trial.get("candidate_family"),
            "source_candidate_id": candidate_id,
            "source_plan_path": _portable_path(source_plan),
            "plan_path": _portable_path(archived_plan),
            "plan_hash": plan_hash,
            "source_p3_result_path": _portable_path(source_p3_result),
            "source_p3_result_hash": _hash_file(source_p3_result),
            "source_p3_makespan": p3_result.get("makespan"),
            "p2_baseline_makespan": int(row["p2_baseline_makespan"]),
            "p3_warm_makespan": int(row["p2_to_p3_warm_makespan"]),
            "p3_mixed_makespan": int(row["p2_to_p3_mixed_makespan"]),
            **fingerprints_by_case[case],
        }
        try:
            evaluated = evaluator.evaluate_multicore(graph.path, plan, 2, tag)
            result_path = results_dir / "raw" / f"{Evaluator._safe_tag(tag)}.json"
            p2_makespan = evaluated.result.get("makespan")
            p2_baseline = record["p2_baseline_makespan"]
            record.update({
                "p2_status": "evaluated",
                "p2_makespan": p2_makespan,
                "p2_delta_vs_baseline": p2_makespan - p2_baseline,
                "p2_relative_change_vs_baseline": (
                    p2_makespan / p2_baseline - 1 if p2_baseline else None
                ),
                "p2_raw_result_path": _portable_path(result_path),
                "p2_raw_result_hash": _hash_file(result_path),
                "evaluation_wall_time_sec": evaluated.elapsed_sec,
                "error": None,
            })
        except EvaluationError as error:
            record.update({
                "p2_status": "evaluation_failed",
                "p2_makespan": None,
                "p2_delta_vs_baseline": None,
                "p2_relative_change_vs_baseline": None,
                "p2_raw_result_path": None,
                "p2_raw_result_hash": None,
                "evaluation_wall_time_sec": error.elapsed_sec,
                "error": str(error),
            })
        records.append(record)
        trial_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))

    (results_dir / "candidate_trials.jsonl").write_text(
        "\n".join(trial_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(results_dir / "reverse_validation.json", {
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.run_id,
        "source_run_id": source_metadata.get("run_id"),
        "evaluation_problem": 2,
        "records": records,
    })
    with (results_dir / "reverse_validation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"run_id={args.run_id} output={results_dir / 'reverse_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())