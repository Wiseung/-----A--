from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FAILURE_STATUSES = {"failure", "timeout", "invalid", "evaluation_failed", "no_legal_candidate"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def formal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row) for row in rows
        if str(row.get("case", "")).startswith("case_")
    ]


def _planned_matrix(
    results_dir: Path, case_manifest_path: Path | None
) -> tuple[list[str], list[int], list[int], str]:
    if case_manifest_path is not None:
        manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
        source = case_manifest_path.name
    else:
        batch_path = results_dir / "batch_manifest.json"
        if batch_path.is_file():
            manifest = json.loads(batch_path.read_text(encoding="utf-8"))
            source = batch_path.name
        else:
            cases = sorted(path.stem for path in (ROOT / "2026_official" / "data").glob("case_*.json"))
            return cases, [1, 2, 3], [1, 2, 3, 4, 5], "official_data_directory"
    cases = manifest.get("cases", manifest.get("case_ids", []))
    problems = manifest.get("problems", [1, 2, 3])
    ncores = manifest.get("ncores", [1, 2, 3, 4, 5])
    if not isinstance(cases, list) or not cases:
        raise ValueError("case manifest must contain a non-empty cases list")
    return sorted(set(cases)), sorted(set(problems)), sorted(set(ncores)), source


def _choose_row(previous: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return candidate
    if candidate.get("status") == "evaluated" and previous.get("status") != "evaluated":
        return candidate
    if previous.get("status") == "evaluated" and candidate.get("status") == "evaluated":
        candidate_objective = (
            candidate.get("makespan", float("inf")),
            candidate.get("added_copy_bytes", float("inf")),
        )
        previous_objective = (
            previous.get("makespan", float("inf")),
            previous.get("added_copy_bytes", float("inf")),
        )
        return candidate if candidate_objective < previous_objective else previous
    return candidate


def collect(
    results_dir: Path,
    case_manifest_path: Path | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = results_dir / "summary.json"
    stored = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
    rows = formal_rows(stored)
    trial_path = results_dir / "candidate_trials.jsonl"
    trials = []
    if trial_path.is_file():
        for line in trial_path.read_text(encoding="utf-8").splitlines():
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    failure_path = results_dir / "process_failures.jsonl"
    process_failures = []
    if failure_path.is_file():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            try:
                process_failures.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cases, problems, ncores_values, manifest_source = _planned_matrix(
        results_dir, case_manifest_path
    )
    if experiment_id is not None:
        rows = [row for row in rows if row.get("experiment_id") == experiment_id]
    if run_id is not None:
        rows = [row for row in rows if row.get("run_id") == run_id]

    runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        identity = (row.get("experiment_id") or "legacy", row.get("run_id") or "legacy")
        runs.setdefault(identity, []).append(row)
    if not runs:
        runs[(experiment_id or "no_results", run_id or "no_results")] = []

    aggregate: list[dict[str, Any]] = []
    manifest_runs = []
    for (current_experiment, current_run), run_rows in sorted(runs.items()):
        by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
        baselines: dict[str, Any] = {}
        trial_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        process_failure_by_key = {}
        for trial in trials:
            if ((trial.get("experiment_id") or "legacy") == current_experiment
                    and (trial.get("run_id") or "legacy") == current_run):
                key = (trial["case"], trial["problem"], trial["ncores"])
                trial_by_key.setdefault(key, []).append(trial)
        for failure in process_failures:
            if ((failure.get("experiment_id") or "legacy") == current_experiment
                    and (failure.get("run_id") or "legacy") == current_run):
                key = (failure["case"], failure["problem"], failure["ncores"])
                process_failure_by_key[key] = failure
        for row in run_rows:
            key = (row["case"], row["problem"], row["ncores"])
            by_key[key] = _choose_row(by_key.get(key), row)
            baseline = row.get("single_core_baseline")
            if baseline is not None:
                baselines.setdefault(row["case"], baseline)

        outcomes = []
        outcome_by_key: dict[tuple[str, int, int], str] = {}
        for case in cases:
            for problem in problems:
                for ncores in ncores_values:
                    key = (case, problem, ncores)
                    row = by_key.get(key)
                    process_failure = process_failure_by_key.get(key)
                    active_process_failure = (
                        process_failure is not None
                        and not (row and row.get("status") == "evaluated")
                    )
                    if active_process_failure:
                        status = "evaluation_failed"
                    elif row is None:
                        candidate_statuses = [
                            trial.get("status") for trial in trial_by_key.get(key, [])
                        ]
                        reasons = [
                            str(trial.get("rejection_reason", "")).lower()
                            for trial in trial_by_key.get(key, [])
                        ]
                        if "evaluation_failed" in candidate_statuses:
                            status = (
                                "timeout" if any("timeout" in reason for reason in reasons)
                                else "evaluation_failed"
                            )
                        elif candidate_statuses and all(
                            candidate_status == "invalid"
                            for candidate_status in candidate_statuses
                        ):
                            status = "invalid"
                        else:
                            status = "not_run"
                    elif row.get("status") == "evaluated" and row.get("makespan") is not None:
                        status = "success"
                    elif row.get("status") in {
                        "timeout", "invalid", "evaluation_failed", "no_legal_candidate"
                    }:
                        status = row["status"]
                    else:
                        status = "not_run" if row.get("status") == "not_run" else "failure"
                    outcome_by_key[key] = status
                    outcomes.append({
                        "case": case,
                        "problem": problem,
                        "ncores": ncores,
                        "status": status,
                        "row_status": row.get("status") if row else None,
                        "makespan": row.get("makespan") if row else None,
                        "error": (
                            process_failure.get("stderr") if active_process_failure
                            else row.get("error") if row else None
                        ),
                        "failure_stage": "process" if active_process_failure else None,
                        "return_code": (
                            process_failure.get("return_code")
                            if active_process_failure else None
                        ),
                    })

        total_planned = len(cases) * len(problems) * len(ncores_values)
        success_total = sum(value == "success" for value in outcome_by_key.values())
        failure_total = sum(value in FAILURE_STATUSES for value in outcome_by_key.values())
        not_run_total = sum(value == "not_run" for value in outcome_by_key.values())
        manifest_runs.append({
            "experiment_id": current_experiment,
            "run_id": current_run,
            "manifest_source": manifest_source,
            "cases": cases,
            "planned_problems": problems,
            "planned_ncores": ncores_values,
            "planned_combinations": total_planned,
            "main_multicore_planned_combinations": (
                len(cases) * len(problems) * sum(ncore > 1 for ncore in ncores_values)
            ),
            "success_combinations": success_total,
            "failure_combinations": failure_total,
            "not_run_combinations": not_run_total,
            "single_core_reference_missing_cases": sum(
                baselines.get(case) is None for case in cases
            ),
            "outcomes": outcomes,
        })

        def planned_status_counts(problem: int, ncores: int) -> dict[str, int]:
            if problem not in problems or ncores not in ncores_values:
                return {
                    "planned_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "not_run_count": 0,
                }
            statuses = [outcome_by_key.get((case, problem, ncores), "not_run") for case in cases]
            return {
                "planned_count": len(cases),
                "success_count": sum(status == "success" for status in statuses),
                "failure_count": sum(status in FAILURE_STATUSES for status in statuses),
                "not_run_count": sum(status == "not_run" for status in statuses),
            }

        def append_speedup_series(
            output_problem: int,
            source_problem: int,
            ncores: int,
            series: str,
            unit_speedup_for_baseline: bool = False,
        ) -> set[str]:
            values = []
            included = []
            for case in cases:
                if unit_speedup_for_baseline and ncores == 1:
                    baseline = baselines.get(case)
                    if baseline is not None:
                        values.append(1.0)
                        included.append(case)
                    continue
                row = by_key.get((case, source_problem, ncores))
                baseline = (row.get("single_core_baseline") if row else None)
                if baseline is None:
                    baseline = baselines.get(case)
                makespan = row.get("makespan") if row else None
                if (row and row.get("status") == "evaluated"
                        and baseline not in (None, 0)
                        and makespan not in (None, 0)):
                    values.append(baseline / makespan)
                    included.append(case)
            counts = planned_status_counts(source_problem, ncores)
            aggregate.append({
                "experiment_id": current_experiment,
                "run_id": current_run,
                "problem": output_problem,
                "source_problem": source_problem,
                "ncores": ncores,
                "series": series,
                "case_count": len(values),
                "missing_cases": len(cases) - len(values),
                "mean_speedup": statistics.mean(values) if values else None,
                "case_set": included,
                **counts,
            })
            return set(included)

        for problem in (1, 2):
            for ncores in range(1, 6):
                append_speedup_series(
                    problem, problem, ncores, "no_l2",
                    unit_speedup_for_baseline=(problem == 1),
                )

        for ncores in range(1, 6):
            no_l2_cases = append_speedup_series(3, 2, ncores, "no_l2")
            l2_cases = append_speedup_series(3, 3, ncores, "l2")
            common = sorted(no_l2_cases & l2_cases)
            cache_ratios = []
            cache_cases = []
            for case in common:
                no_l2 = by_key.get((case, 2, ncores), {}).get("makespan")
                l2 = by_key.get((case, 3, ncores), {}).get("makespan")
                if no_l2 not in (None, 0) and l2 not in (None, 0):
                    cache_ratios.append(no_l2 / l2)
                    cache_cases.append(case)
            planned_pair = (2 in problems and 3 in problems
                            and ncores in ncores_values)
            pair_status = [
                (outcome_by_key.get((case, 2, ncores)),
                 outcome_by_key.get((case, 3, ncores)))
                for case in cases
            ] if planned_pair else []
            aggregate.append({
                "experiment_id": current_experiment,
                "run_id": current_run,
                "problem": 3,
                "source_problem": "2_vs_3",
                "ncores": ncores,
                "series": "cache_speedup",
                "case_count": len(cache_ratios),
                "missing_cases": len(cases) - len(cache_ratios),
                "mean_speedup": statistics.mean(cache_ratios) if cache_ratios else None,
                "case_set": cache_cases,
                "common_case_set": common,
                "planned_count": len(cases) if planned_pair else 0,
                "success_count": sum(a == "success" and b == "success" for a, b in pair_status),
                "failure_count": sum(
                    a in FAILURE_STATUSES or b in FAILURE_STATUSES
                    for a, b in pair_status
                ),
                "not_run_count": sum(a == "not_run" or b == "not_run" for a, b in pair_status),
            })

    _write_json(results_dir / "complete_case_manifest.json", {
        "manifest_source": manifest_source,
        "runs": manifest_runs,
    })
    _write_json(results_dir / "summary_by_run.json", {
        "runs": [
            {
                **{
                    key: value for key, value in run.items()
                    if key != "outcomes"
                },
                "aggregate": [
                    row for row in aggregate
                    if row["experiment_id"] == run["experiment_id"]
                    and row["run_id"] == run["run_id"]
                ],
            }
            for run in manifest_runs
        ],
    })
    _write_json(results_dir / "aggregate_summary.json", aggregate)
    with (results_dir / "aggregate_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = [
            "experiment_id", "run_id", "problem", "source_problem", "ncores",
            "series", "case_count", "missing_cases", "planned_count",
            "success_count", "failure_count", "not_run_count", "mean_speedup",
            "case_set", "common_case_set",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in aggregate:
            csv_row = dict(row)
            csv_row["case_set"] = json.dumps(row.get("case_set", []), ensure_ascii=False)
            csv_row["common_case_set"] = json.dumps(
                row.get("common_case_set", []), ensure_ascii=False
            )
            writer.writerow(csv_row)
    return rows, aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate official evaluator results against the planned case matrix.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    rows, aggregate = collect(
        args.results_dir,
        args.case_manifest,
        args.experiment_id,
        args.run_id,
    )
    cases, _, _, _ = _planned_matrix(args.results_dir, args.case_manifest)
    print(f"manifest_cases={len(cases)}; result_rows={len(rows)}")
    for row in aggregate:
        if row["mean_speedup"] is not None:
            print(
                f"p{row['problem']} {row['series']} n{row['ncores']}: "
                f"mean_speedup={row['mean_speedup']:.6g} "
                f"cases={row['case_count']} missing={row['missing_cases']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())