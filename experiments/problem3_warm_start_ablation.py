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
from solver.graph_io import Graph, load_config, load_graph
from solver.legality import validate_plan
from solver.run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)
from solver.solver import plan_signature, solve_one


EXPERIMENT_ID = "r02_problem3_warm_start"
ARMS = (
    "p3_original",
    "p2_baseline",
    "p2_to_p3_warm",
    "p2_to_p3_mixed",
)
CSV_FIELDS = [
    "case",
    "ncores",
    *[
        f"{arm}_{field}"
        for arm in ARMS
        for field in (
            "status",
            "makespan",
            "best_candidate_source",
            "candidate_id",
            "candidate_family",
            "source_plan",
            "plan_hash",
            "added_copy_bytes",
            "cache_hit_rate",
            "evaluator_calls",
        )
    ],
    "p2_warm_over_p3_original",
    "p2_warm_over_p2_baseline",
    "mixed_over_p2_warm",
    "warm_not_worse_than_p2_baseline",
    "mixed_not_worse_than_p2_baseline",
]


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


def _append_trial(path: Path, trial: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(trial, ensure_ascii=False, sort_keys=True) + "\n")


def _read_candidate_trial(results_dir: Path, candidate_id: str | None) -> dict[str, Any]:
    if candidate_id is None:
        return {}
    path = results_dir / "candidate_trials.jsonl"
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            trial = json.loads(line)
        except json.JSONDecodeError:
            continue
        if trial.get("candidate_id") == candidate_id:
            return trial
    return {}


def _solver_arm(
    name: str,
    row: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    trial = _read_candidate_trial(results_dir, row.get("candidate_id"))
    candidate_id = row.get("candidate_id")
    plan_path = (
        results_dir / "schedules" / f"{candidate_id}.json"
        if candidate_id else None
    )
    plan_hash = None
    if plan_path is not None and plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plan = None
        if isinstance(plan, dict):
            plan_hash = _plan_hash(plan)
    return {
        "arm": name,
        "status": row.get("status"),
        "makespan": row.get("makespan"),
        "best_candidate_source": row.get("best_candidate_source"),
        "candidate_id": row.get("candidate_id"),
        "candidate_family": trial.get("candidate_family"),
        "parent_candidate": trial.get("parent_candidate"),
        "source_plan": trial.get("source_plan"),
        "plan_path": _portable_path(plan_path) if plan_hash and plan_path else None,
        "plan_hash": plan_hash,
        "added_copy_bytes": row.get("added_copy_bytes"),
        "cache_hit_rate": row.get("cache_hit_rate"),
        "evaluator_calls": row.get("evaluator_calls"),
        "evaluator_runtime_sec": row.get("candidate_eval_wall_sec"),
        "error": row.get("error"),
    }


def _fixed_p3_arm(
    case: str,
    ncores: int,
    plan: dict[str, Any],
    source_path: Path,
    archived_path: Path,
    graph_path: Path,
    fingerprints: dict[str, str | None],
    evaluator: Evaluator,
    results_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    tag = f"{case}_n{ncores}_p3_original_{run_id}"
    record: dict[str, Any] = {
        "arm": "p3_original",
        "case": case,
        "ncores": ncores,
        "candidate_id": tag,
        "candidate_family": "p3_original_history",
        "best_candidate_source": "p3_original_history",
        "parent_candidate": None,
        "source_plan": _portable_path(source_path),
        "archived_plan": _portable_path(archived_path),
        "plan_hash": _plan_hash(plan),
        **fingerprints,
    }
    try:
        evaluated = evaluator.evaluate_multicore(graph_path, plan, 3, tag)
        movement = evaluated.result.get("data_movement_bytes", {})
        record.update({
            "status": "evaluated",
            "makespan": evaluated.result.get("makespan"),
            "added_copy_bytes": movement.get("added_copy_bytes"),
            "cache_hit_rate": evaluated.result.get("cache_stats", {}).get(
                "hit_rate"
            ),
            "evaluation_wall_time_sec": evaluated.elapsed_sec,
        })
    except EvaluationError as error:
        record.update({
            "status": "evaluation_failed",
            "makespan": None,
            "added_copy_bytes": None,
            "cache_hit_rate": None,
            "evaluation_wall_time_sec": error.elapsed_sec,
            "error": str(error),
        })
    _append_trial(results_dir / "candidate_trials.jsonl", {
        **record,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "problem": 3,
        "status": record["status"],
        "official_makespan": record.get("makespan"),
        "accepted": None,
    })
    return record


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _write_outputs(results_dir: Path, state: dict[str, Any]) -> None:
    state["comparisons"].sort(key=lambda row: (row["case"], row["ncores"]))
    _write_json(results_dir / "problem3_warm_start_ablation.json", state)
    rows = []
    for comparison in state["comparisons"]:
        row = {
            "case": comparison["case"],
            "ncores": comparison["ncores"],
        }
        for arm in ARMS:
            arm_row = comparison.get(arm, {})
            row.update({
                f"{arm}_{field}": arm_row.get(field)
                for field in (
                    "status",
                    "makespan",
                    "best_candidate_source",
                    "candidate_id",
                    "candidate_family",
                    "source_plan",
                    "plan_hash",
                    "added_copy_bytes",
                    "cache_hit_rate",
                    "evaluator_calls",
                )
            })
        ratios = comparison.get("ratios", {})
        row.update({
            "p2_warm_over_p3_original": ratios.get("p2_warm_over_p3_original"),
            "p2_warm_over_p2_baseline": ratios.get("p2_warm_over_p2_baseline"),
            "mixed_over_p2_warm": ratios.get("mixed_over_p2_warm"),
            "warm_not_worse_than_p2_baseline": ratios.get(
                "warm_not_worse_than_p2_baseline"
            ),
            "mixed_not_worse_than_p2_baseline": ratios.get(
                "mixed_not_worse_than_p2_baseline"
            ),
        })
        rows.append(row)
    csv_path = results_dir / "problem3_warm_start_ablation.csv"
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def _solve_arm(
    name: str,
    graph: Graph,
    problem: int,
    ncores: int,
    config: dict[str, Any],
    evaluator: Evaluator,
    results_dir: Path,
    history_dir: Path,
    run_id: str,
    max_evals: int,
    time_limit: float,
    baseline_timeout: float,
    seed: int,
) -> dict[str, Any]:
    row = solve_one(
        graph,
        problem,
        ncores,
        config,
        evaluator,
        results_dir,
        time_limit,
        max_evals,
        seed,
        False,
        baseline_timeout,
        False,
        EXPERIMENT_ID,
        run_id,
        history_dir,
    )
    return _solver_arm(name, row, results_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare historical Problem 3, P2 warm-start, and mixed candidates."
    )
    parser.add_argument("--cases", nargs="+", default=["case_019", "case_034", "case_014"])
    parser.add_argument("--ncores", nargs="+", type=int, default=[2, 4, 5])
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--history-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--baseline-timeout", type=float, default=2.0)
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument("--retain-traces", action="store_true")
    parser.add_argument("--p2-max-evals", type=int, default=2)
    parser.add_argument("--mixed-max-evals", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (args.time_limit <= 0 or args.baseline_timeout <= 0
            or args.evaluator_timeout <= 0 or args.p2_max_evals < 1
            or args.mixed_max_evals < 2):
        parser.error("timeouts and budgets must be positive; mixed-max-evals must be at least 2")
    if not args.ncores or any(not 2 <= ncore <= 5 for ncore in args.ncores):
        parser.error("ncores values must be between 2 and 5")
    if not args.cases:
        parser.error("at least one case is required")
    if len(set(args.cases)) != len(args.cases) or len(set(args.ncores)) != len(args.ncores):
        parser.error("cases and ncores must not contain duplicates")
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")

    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")
    official_root = args.official_root.resolve()
    source_results_dir = args.source_results_dir.resolve()
    history_dir = args.history_dir.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else ROOT / "results_round2" / EXPERIMENT_ID / run_id
    )
    data_dir = official_root / "data"
    available_cases = {path.stem: path for path in data_dir.glob("case_*.json")}
    missing = set(args.cases) - available_cases.keys()
    if missing:
        parser.error(f"unknown cases: {sorted(missing)}")

    config = load_config(official_root, args.config)
    graphs = {case: load_graph(available_cases[case]) for case in args.cases}
    fingerprints_by_case = {
        case: build_fingerprints(
            graphs[case].path,
            config["path"],
            official_root,
            ROOT / "solver",
        )
        for case in args.cases
    }
    original_plans: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for case, graph in graphs.items():
        for ncores in args.ncores:
            source_path = source_results_dir / "schedules" / (
                f"{case}_p3_n{ncores}_best.json"
            )
            if not source_path.is_file():
                parser.error(f"missing historical Problem 3 plan: {source_path}")
            try:
                plan = json.loads(source_path.read_text(encoding="utf-8"))
                validate_plan(graph, plan, ncores)
            except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
                parser.error(f"invalid historical plan {source_path}: {error}")
            original_plans[(case, ncores)] = (source_path, plan)

    common_fingerprints = fingerprints_by_case[args.cases[0]]
    run_metadata = make_run_metadata(
        EXPERIMENT_ID,
        run_id,
        common_fingerprints,
        {
            "cases": list(args.cases),
            "ncores": list(args.ncores),
            "source_results_dir": str(source_results_dir),
            "history_dir": str(history_dir),
            "graph_hashes": {
                case: fingerprints["graph_hash"]
                for case, fingerprints in fingerprints_by_case.items()
            },
            "source_p3_plan_hashes": {
                f"{case}:n{ncores}": _plan_hash(plan)
                for (case, ncores), (_, plan) in original_plans.items()
            },
            "time_limit_sec": args.time_limit,
            "baseline_timeout_sec": args.baseline_timeout,
            "evaluator_timeout_sec": args.evaluator_timeout,
            "p2_max_evals": args.p2_max_evals,
            "warm_max_evals": 1,
            "mixed_max_evals": args.mixed_max_evals,
            "seed": args.seed,
            "improve_enabled": False,
            "retain_traces": args.retain_traces,
        },
    )
    batch_manifest = {
        "cases": list(args.cases),
        "problems": [2, 3],
        "ncores": list(args.ncores),
        "arms": list(ARMS),
        "data_dir": _portable_path(data_dir),
    }
    metadata_path = results_dir / "run_metadata.json"
    manifest_path = results_dir / "batch_manifest.json"
    if args.resume:
        if not metadata_path.is_file() or not manifest_path.is_file():
            parser.error("cannot resume: run metadata or batch manifest is missing")
        try:
            ensure_run_metadata(metadata_path, run_metadata)
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))
        if existing_manifest != batch_manifest:
            parser.error("resume matrix does not match the original batch manifest")
    else:
        if results_dir.exists() and any(results_dir.iterdir()):
            parser.error("results directory is not empty; use --resume with the same --run-id")
        ensure_run_metadata(metadata_path, run_metadata)
        _write_json(manifest_path, batch_manifest)

    output_path = results_dir / "problem3_warm_start_ablation.json"
    state: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "cases": [],
        "comparisons": [],
    }
    if args.resume and output_path.is_file():
        try:
            state = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid experiment output: {error}")
        if state.get("experiment_id") != EXPERIMENT_ID or state.get("run_id") != run_id:
            parser.error("experiment output does not match run identity")

    evaluator = Evaluator(
        official_root,
        config["path"],
        results_dir,
        args.evaluator_timeout,
        retain_traces=args.retain_traces,
    )
    total = len(args.cases) * len(args.ncores)
    done = 0
    for case in args.cases:
        graph = graphs[case]
        fingerprints = fingerprints_by_case[case]
        for ncores in args.ncores:
            done += 1
            comparison = next((
                row for row in state["comparisons"]
                if row.get("case") == case and row.get("ncores") == ncores
            ), None)
            if comparison is None:
                comparison = {"case": case, "ncores": ncores}
                state["comparisons"].append(comparison)
            print(f"[{done}/{total}] {case} n{ncores}")

            if "p3_original" not in comparison:
                source_path, plan = original_plans[(case, ncores)]
                archived_path = results_dir / "fixed_plans" / (
                    f"{case}_n{ncores}_p3_original_{_plan_hash(plan)[:12]}.json"
                )
                _write_json(archived_path, plan)
                comparison["p3_original"] = _fixed_p3_arm(
                    case,
                    ncores,
                    plan,
                    source_path,
                    archived_path,
                    graph.path,
                    fingerprints,
                    evaluator,
                    results_dir,
                    run_id,
                )
                _write_outputs(results_dir, state)

            if "p2_baseline" not in comparison:
                comparison["p2_baseline"] = _solve_arm(
                    "p2_baseline",
                    graph,
                    2,
                    ncores,
                    config,
                    evaluator,
                    results_dir,
                    history_dir,
                    run_id,
                    args.p2_max_evals,
                    args.time_limit,
                    args.baseline_timeout,
                    args.seed,
                )
                _write_outputs(results_dir, state)

            if "p2_to_p3_warm" not in comparison:
                comparison["p2_to_p3_warm"] = _solve_arm(
                    "p2_to_p3_warm",
                    graph,
                    3,
                    ncores,
                    config,
                    evaluator,
                    results_dir,
                    history_dir,
                    run_id,
                    1,
                    args.time_limit,
                    args.baseline_timeout,
                    args.seed,
                )
                _write_outputs(results_dir, state)

            if "p2_to_p3_mixed" not in comparison:
                comparison["p2_to_p3_mixed"] = _solve_arm(
                    "p2_to_p3_mixed",
                    graph,
                    3,
                    ncores,
                    config,
                    evaluator,
                    results_dir,
                    history_dir,
                    run_id,
                    args.mixed_max_evals,
                    args.time_limit,
                    args.baseline_timeout,
                    args.seed,
                )
                warm = comparison["p2_to_p3_warm"].get("makespan")
                baseline = comparison["p2_baseline"].get("makespan")
                mixed = comparison["p2_to_p3_mixed"].get("makespan")
                comparison["ratios"] = {
                    "p2_warm_over_p3_original": _ratio(
                        comparison["p2_to_p3_warm"].get("makespan"),
                        comparison["p3_original"].get("makespan"),
                    ),
                    "p2_warm_over_p2_baseline": _ratio(warm, baseline),
                    "mixed_over_p2_warm": _ratio(mixed, warm),
                    "warm_not_worse_than_p2_baseline": (
                        warm <= baseline
                        if isinstance(warm, (int, float))
                        and isinstance(baseline, (int, float))
                        else None
                    ),
                    "mixed_not_worse_than_p2_baseline": (
                        mixed <= baseline
                        if isinstance(mixed, (int, float))
                        and isinstance(baseline, (int, float))
                        else None
                    ),
                    "ratio_convention": "numerator_makespan / denominator_makespan",
                }
                _write_outputs(results_dir, state)

            for arm in ARMS:
                result = comparison.get(arm, {})
                print(
                    f"  {arm}: status={result.get('status')} "
                    f"makespan={result.get('makespan')} "
                    f"best={result.get('best_candidate_source')}"
                )

    output_path = results_dir / "problem3_warm_start_ablation.json"
    _write_outputs(results_dir, state)
    print(f"run_id={run_id} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
