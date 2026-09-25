from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_STATUSES = {"success", "equivalent_plan"}
BASELINE_ARM_BY_PROBLEM = {2: "contiguous_p2", 3: "contiguous_p3"}
CANDIDATE_ARMS = ("cagg_lite", "cagg_lite_coverage")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _objective(row: dict[str, Any]) -> tuple[float, float, float, float]:
    def value(name: str) -> float:
        item = row.get(name)
        return float(item) if item is not None else float("inf")

    cache_hit_bytes = row.get("cache_hit_bytes")
    cache_key = -float(cache_hit_bytes) if cache_hit_bytes is not None else 0.0
    return (
        value("makespan"),
        value("added_copy_bytes"),
        value("spill_added_copy_bytes"),
        cache_key,
    )


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("status") in SUCCESS_STATUSES]
    if not valid:
        return None
    return min(
        valid,
        key=lambda row: (_objective(row), row.get("group_count", 0)),
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _comparison_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["case"], row["ncores"], row["evaluation_problem"]


def evaluate_promotion(
    summary: dict[str, Any],
    collapse_cases: list[str] | None = None,
    min_wins: int = 1,
    min_win_cases: int = 2,
    max_regression_ratio: float = 0.0,
) -> dict[str, Any]:
    matrix_spec = summary.get("matrix_spec", {})
    cases = list(matrix_spec.get("cases", []))
    ncores = list(matrix_spec.get("ncores", []))
    evaluation_problems = list(matrix_spec.get("evaluation_problems", [2, 3]))
    collapse_cases = collapse_cases or ["case_019", "case_028"]
    rows = summary.get("comparisons", [])
    expected_keys = [
        (case, core_count, problem)
        for case in cases
        for core_count in ncores
        for problem in evaluation_problems
    ]
    by_arm: dict[tuple[str, tuple[str, int, int]], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("arm") not in (*BASELINE_ARM_BY_PROBLEM.values(), *CANDIDATE_ARMS):
            continue
        by_arm.setdefault((row["arm"], _comparison_key(row)), []).append(row)

    candidates = []
    for arm in CANDIDATE_ARMS:
        comparisons = []
        missing = []
        collapse_violations = []
        for key in expected_keys:
            case, core_count, problem = key
            candidate = _best_row(by_arm.get((arm, key), []))
            baseline_arm = BASELINE_ARM_BY_PROBLEM.get(problem)
            baseline = _best_row(by_arm.get((baseline_arm, key), [])) if baseline_arm else None
            if candidate is None or baseline is None:
                missing.append({
                    "case": case,
                    "ncores": core_count,
                    "evaluation_problem": problem,
                    "missing_candidate": candidate is None,
                    "missing_baseline": baseline is None,
                })
                continue
            candidate_makespan = float(candidate["makespan"])
            baseline_makespan = float(baseline["makespan"])
            delta_ratio = (
                candidate_makespan / baseline_makespan - 1.0
                if baseline_makespan else None
            )
            candidate_objective = _objective(candidate)
            baseline_objective = _objective(baseline)
            if candidate_objective < baseline_objective:
                outcome = "win"
            elif candidate_objective == baseline_objective:
                outcome = "tie"
            else:
                outcome = "loss"
            comparison = {
                "case": case,
                "ncores": core_count,
                "evaluation_problem": problem,
                "candidate_group_count": candidate.get("group_count"),
                "baseline_arm": baseline_arm,
                "baseline_group_count": baseline.get("group_count"),
                "candidate_makespan": candidate.get("makespan"),
                "baseline_makespan": baseline.get("makespan"),
                "makespan_delta_ratio": delta_ratio,
                "candidate_added_copy_bytes": candidate.get("added_copy_bytes"),
                "baseline_added_copy_bytes": baseline.get("added_copy_bytes"),
                "candidate_spill_added_copy_bytes": candidate.get(
                    "spill_added_copy_bytes"
                ),
                "baseline_spill_added_copy_bytes": baseline.get(
                    "spill_added_copy_bytes"
                ),
                "active_core_count": candidate.get("active_core_count"),
                "outcome": outcome,
            }
            comparisons.append(comparison)
            if (
                case in collapse_cases
                and core_count >= 4
                and candidate.get("active_core_count") is not None
                and candidate["active_core_count"] < core_count - 1
            ):
                collapse_violations.append({
                    "case": case,
                    "ncores": core_count,
                    "active_core_count": candidate["active_core_count"],
                    "minimum_allowed": core_count - 1,
                    "group_count": candidate.get("group_count"),
                })

        deltas = [
            comparison["makespan_delta_ratio"]
            for comparison in comparisons
            if comparison["makespan_delta_ratio"] is not None
        ]
        wins = [item for item in comparisons if item["outcome"] == "win"]
        ties = [item for item in comparisons if item["outcome"] == "tie"]
        losses = [item for item in comparisons if item["outcome"] == "loss"]
        p2_losses = [item for item in losses if item["evaluation_problem"] == 2]
        win_cases = sorted({item["case"] for item in wins})
        missing_baseline_or_candidate = bool(missing)
        max_delta = max(deltas, default=None)
        p90_delta = _percentile(deltas, 0.90)
        median_delta = statistics.median(deltas) if deltas else None
        spill_deltas = [
            float(item["candidate_spill_added_copy_bytes"])
            - float(item["baseline_spill_added_copy_bytes"])
            for item in comparisons
            if item["candidate_spill_added_copy_bytes"] is not None
            and item["baseline_spill_added_copy_bytes"] is not None
        ]
        gates = {
            "complete_comparisons": not missing_baseline_or_candidate,
            "no_problem2_regression": not p2_losses,
            "no_severe_core_collapse": not collapse_violations,
            "worst_regression_within_limit": (
                max_delta is not None and max_delta <= max_regression_ratio
            ),
            "minimum_wins": len(wins) >= min_wins,
            "wins_across_cases": len(win_cases) >= min_win_cases,
        }
        candidates.append({
            "arm": arm,
            "comparison_count": len(comparisons),
            "expected_comparison_count": len(expected_keys),
            "missing": missing,
            "wins": len(wins),
            "ties": len(ties),
            "losses": len(losses),
            "problem2_losses": len(p2_losses),
            "win_cases": win_cases,
            "median_makespan_delta_ratio": median_delta,
            "p90_makespan_delta_ratio": p90_delta,
            "worst_makespan_delta_ratio": max_delta,
            "median_spill_delta": (
                statistics.median(spill_deltas) if spill_deltas else None
            ),
            "worst_spill_delta": max(spill_deltas, default=None),
            "collapse_violations": collapse_violations,
            "gates": gates,
            "ready": all(gates.values()),
            "comparisons": comparisons,
        })

    ready = [candidate for candidate in candidates if candidate["ready"]]
    recommended = None
    if ready:
        recommended = min(
            ready,
            key=lambda candidate: (
                candidate["median_makespan_delta_ratio"],
                candidate["median_spill_delta"]
                if candidate["median_spill_delta"] is not None else float("inf"),
                -candidate["wins"],
                candidate["arm"],
            ),
        )["arm"]
    return {
        "status": "ready" if recommended is not None else "not_ready",
        "recommended_arm": recommended,
        "criteria": {
            "collapse_cases": collapse_cases,
            "min_wins": min_wins,
            "min_win_cases": min_win_cases,
            "max_regression_ratio": max_regression_ratio,
            "baseline_arm_by_problem": BASELINE_ARM_BY_PROBLEM,
        },
        "matrix_spec": matrix_spec,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 4 candidate promotion gates."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collapse-cases", nargs="+", default=["case_019", "case_028"])
    parser.add_argument("--min-wins", type=int, default=1)
    parser.add_argument("--min-win-cases", type=int, default=2)
    parser.add_argument("--max-regression-ratio", type=float, default=0.0)
    args = parser.parse_args()
    if args.min_wins < 0 or args.min_win_cases < 0:
        parser.error("minimum win thresholds must be non-negative")
    if args.max_regression_ratio < 0:
        parser.error("max-regression-ratio must be non-negative")
    summary_path = args.results_dir.resolve() / "phase4_matrix.json"
    if not summary_path.is_file():
        parser.error(f"missing phase4_matrix.json: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read phase4 summary: {error}")
    result = evaluate_promotion(
        summary,
        collapse_cases=args.collapse_cases,
        min_wins=args.min_wins,
        min_win_cases=args.min_win_cases,
        max_regression_ratio=args.max_regression_ratio,
    )
    output = args.output.resolve() if args.output else args.results_dir.resolve() / "promotion_summary.json"
    _write_json(output, result)
    print(
        f"status={result['status']} recommended_arm={result['recommended_arm']} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())