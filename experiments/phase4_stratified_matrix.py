from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.evaluate_adapter import EvaluationError, Evaluator
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


EXPERIMENT_ID = "r04_stratified_matrix"
CASES = [
    "case_001", "case_014", "case_016", "case_019",
    "case_020", "case_028", "case_034", "case_093",
]
ARMS = {
    "contiguous_p1": {
        "partition_strategy": "contiguous",
        "partition_problem": 1,
        "placement_scoring": "baseline",
    },
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
    "chain_contiguous": {
        "partition_strategy": "chain_contiguous",
        "partition_problem": 2,
        "placement_scoring": "baseline",
    },
    "chain_contiguous_p1": {
        "partition_strategy": "chain_contiguous",
        "partition_problem": 1,
        "placement_scoring": "baseline",
    },
}
GROUP_COUNTS = [8, 10, 12]
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


def _hash_plan(plan: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(plan_signature(plan).encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _matrix_hash(spec: dict[str, Any]) -> str:
    import hashlib

    serialized = json.dumps(
        spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _candidate_matrix(
    cases: list[str],
    ncores: list[int],
    group_counts: list[int],
    arms: list[str],
    run_id: str,
) -> list[dict[str, Any]]:
    candidates = []
    for case in cases:
        for core_count in ncores:
            for arm in arms:
                for group_count in group_counts:
                    candidates.append({
                        "candidate_id": (
                            f"{case}_n{core_count}_{arm}_g{group_count}_{run_id}"
                        ),
                        "case": case,
                        "ncores": core_count,
                        "arm": arm,
                        "group_count": group_count,
                        **ARMS[arm],
                    })
    return candidates


def _proxy_score(
    metrics: dict[str, Any],
    placement_diagnostics: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, int, int]:
    estimated_finish = max(
        (
            float(item.get("group_finish", 0.0))
            for item in placement_diagnostics.get("groups", [])
        ),
        default=0.0,
    )
    added_copy = placement_diagnostics.get(
        "estimated_partition_added_copy_bytes_recomputed"
    )
    if added_copy is None:
        added_copy = placement_diagnostics.get(
            "estimated_partition_added_copy_bytes", 10**30
        )
    imbalance = max(
        float(metrics.get("m_load_imbalance", 0.0)),
        float(metrics.get("v_load_imbalance", 0.0)),
    )
    boundary_bytes = placement_diagnostics.get(
        "boundary_copy_bytes_est",
        placement_diagnostics.get(
            "boundary_bytes_total",
            placement_diagnostics.get("boundary_tensor_bytes", added_copy),
        ),
    )
    overflow_bytes = (
        int(placement_diagnostics.get("max_l1_overflow_bytes_est", 0))
        + int(placement_diagnostics.get("max_ub_overflow_bytes_est", 0))
    )
    l1_residence = placement_diagnostics.get("l1_residence_bytes_est", {})
    ub_residence = placement_diagnostics.get("ub_residence_bytes_est", {})
    max_l1_residence = max(
        (float(value) for value in l1_residence.values()),
        default=0.0,
    ) if isinstance(l1_residence, dict) else 0.0
    max_ub_residence = max(
        (float(value) for value in ub_residence.values()),
        default=0.0,
    ) if isinstance(ub_residence, dict) else 0.0
    return (
        estimated_finish,
        float(boundary_bytes),
        float(overflow_bytes),
        float(metrics.get("max_core_compute", 0)),
        imbalance,
        max_l1_residence,
        max_ub_residence,
        int(placement_diagnostics.get("estimated_repeated_input_bytes", 0)),
        -int(metrics.get("active_core_count", 0)),
    )


def _rank_proxy_candidates(
    candidates: list[dict[str, Any]],
    proxy_screen: bool,
    max_selected: int = 4,
    boundary_ratio_limit: float = 2.0,
) -> None:
    contiguous: dict[tuple[str, int, int], dict[str, Any]] = {}
    for candidate in candidates:
        if (
            candidate.get("partition_strategy") != "contiguous"
            or candidate.get("proxy_score") is None
        ):
            continue
        key = (candidate["case"], candidate["ncores"], candidate["group_count"])
        previous = contiguous.get(key)
        if previous is None or (
            candidate.get("arm") == "contiguous_p2"
            and previous.get("arm") != "contiguous_p2"
        ):
            contiguous[key] = candidate
    for candidate in candidates:
        key = (candidate["case"], candidate["ncores"], candidate["group_count"])
        baseline = contiguous.get(key)
        if candidate.get("proxy_score") is None:
            candidate["safety_status"] = "invalid"
            candidate["boundary_ratio_to_contiguous"] = None
            candidate["overflow_delta_to_contiguous"] = None
            continue
        if baseline is None or candidate.get("partition_strategy") == "contiguous":
            candidate["safety_status"] = "baseline_or_uncertain"
            candidate["boundary_ratio_to_contiguous"] = None
            candidate["overflow_delta_to_contiguous"] = None
            candidate["gate_decision"] = "retain"
            candidate["gate_reason"] = "incumbent_or_no_contiguous_reference"
            continue
        candidate_boundary = float(candidate.get("boundary_copy_bytes_est", 0))
        baseline_boundary = float(baseline.get("boundary_copy_bytes_est", 0))
        candidate_overflow = (
            int(candidate.get("max_l1_overflow_bytes_est", 0))
            + int(candidate.get("max_ub_overflow_bytes_est", 0))
        )
        baseline_overflow = (
            int(baseline.get("max_l1_overflow_bytes_est", 0))
            + int(baseline.get("max_ub_overflow_bytes_est", 0))
        )
        boundary_ratio = (
            candidate_boundary / baseline_boundary
            if baseline_boundary > 0
            else (float("inf") if candidate_boundary > 0 else 1.0)
        )
        candidate["boundary_ratio_to_contiguous"] = boundary_ratio
        candidate["overflow_delta_to_contiguous"] = candidate_overflow - baseline_overflow
        candidate["boundary_copy_delta_to_contiguous"] = (
            candidate_boundary - baseline_boundary
        )
        candidate["l1_overflow_delta_to_contiguous"] = (
            int(candidate.get("max_l1_overflow_bytes_est", 0))
            - int(baseline.get("max_l1_overflow_bytes_est", 0))
        )
        candidate["ub_overflow_delta_to_contiguous"] = (
            int(candidate.get("max_ub_overflow_bytes_est", 0))
            - int(baseline.get("max_ub_overflow_bytes_est", 0))
        )
        candidate["max_core_compute_delta_to_contiguous"] = (
            float(candidate.get("max_core_compute", 0))
            - float(baseline.get("max_core_compute", 0))
        )
        candidate["m_load_imbalance_delta_to_contiguous"] = (
            float(candidate.get("m_load_imbalance", 0.0))
            - float(baseline.get("m_load_imbalance", 0.0))
        )
        candidate["v_load_imbalance_delta_to_contiguous"] = (
            float(candidate.get("v_load_imbalance", 0.0))
            - float(baseline.get("v_load_imbalance", 0.0))
        )
        candidate["active_core_delta_to_contiguous"] = (
            int(candidate.get("active_core_count", 0))
            - int(baseline.get("active_core_count", 0))
        )
        candidate["critical_path_cross_core_delta_to_contiguous"] = (
            int(candidate.get("critical_path_cross_core_edges", 0))
            - int(baseline.get("critical_path_cross_core_edges", 0))
        )
        extra_boundary = candidate_boundary - baseline_boundary
        residence_reduction = float(baseline.get("long_lived_tensor_bytes", 0)) - float(
            candidate.get("long_lived_tensor_bytes", 0)
        )
        candidate["residence_reduction_per_extra_copy_byte"] = (
            residence_reduction / extra_boundary if extra_boundary > 0 else None
        )
        candidate["overflow_to_boundary_ratio"] = (
            candidate_overflow / candidate_boundary if candidate_boundary > 0 else 0.0
        )
        candidate["safety_status"] = (
            "high_risk"
            if boundary_ratio > boundary_ratio_limit
            and candidate_overflow >= baseline_overflow
            else "safe_or_uncertain"
        )
        candidate["gate_decision"] = "high_risk_reserve" if candidate["safety_status"] == "high_risk" else "retain"
        candidate["gate_reason"] = (
            "boundary_ratio_and_overflow"
            if candidate["safety_status"] == "high_risk"
            else "not_high_risk"
        )

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(
            (candidate["case"], candidate["ncores"], candidate["arm"]),
            [],
        ).append(candidate)
    for group in groups.values():
        valid = sorted(
            (
                item for item in group
                if item.get("proxy_score") is not None
                and item.get("safety_status") != "high_risk"
            ),
            key=lambda item: (
                tuple(item["proxy_score"]),
                item["group_count"],
                item["candidate_id"],
            ),
        )
        def objective(item: dict[str, Any]) -> tuple[float, float, float, float]:
            score = item["proxy_score"]
            return (
                float(score[0]),
                float(score[1]),
                float(score[2]),
                float(score[4]),
            )

        def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
            left_objective = objective(left)
            right_objective = objective(right)
            return (
                all(a <= b for a, b in zip(left_objective, right_objective))
                and any(a < b for a, b in zip(left_objective, right_objective))
            )

        pareto = [
            item for item in valid
            if not any(
                other is not item and dominates(other, item)
                for other in valid
            )
        ]
        selected: list[dict[str, Any]] = []
        reasons: dict[str, list[str]] = {}
        criterion_order = (
            (0, "estimated_finish"),
            (1, "boundary_copy"),
            (2, "overflow"),
            (3, "m_v_balance"),
        )
        if group and group[0].get("partition_strategy") == "contiguous":
            if valid:
                incumbent = min(
                    valid,
                    key=lambda item: (tuple(item["proxy_score"]), item["group_count"], item["candidate_id"]),
                )
                selected = [incumbent]
                reasons[incumbent["candidate_id"]] = ["incumbent"]
        else:
            if valid:
                for index, reason in criterion_order:
                    candidate = min(
                        pareto or valid,
                        key=lambda item: (objective(item)[index], item["group_count"]),
                    )
                    if candidate not in selected:
                        if len(selected) < max_selected:
                            selected.append(candidate)
                            reasons.setdefault(candidate["candidate_id"], []).append(reason)
                for candidate in sorted(pareto, key=lambda item: tuple(item["proxy_score"])):
                    if len(selected) >= max_selected:
                        break
                    if candidate not in selected:
                        selected.append(candidate)
                        reasons.setdefault(candidate["candidate_id"], []).append("pareto")
            risky = [item for item in group if item.get("safety_status") == "high_risk"]
            if risky:
                risk_candidate = min(
                    risky,
                    key=lambda item: (tuple(item["proxy_score"]), item["group_count"], item["candidate_id"]),
                )
                if risk_candidate not in selected:
                    selected.append(risk_candidate)
                    reasons.setdefault(risk_candidate["candidate_id"], []).append("high_risk_reserve")

        for rank, item in enumerate(valid, start=1):
            item["proxy_rank"] = rank
            item["pareto_kept"] = item in pareto
            item["proxy_selection_reasons"] = reasons.get(item["candidate_id"], [])
            item["proxy_selected"] = not proxy_screen or item in selected
            if item["proxy_selected"]:
                item["candidate_status"] = (
                    "proxy_selected_high_risk"
                    if item.get("safety_status") == "high_risk"
                    else "proxy_selected"
                )
            else:
                item["candidate_status"] = (
                    "not_run_high_risk"
                    if item.get("safety_status") == "high_risk"
                    else "screened_out"
                )
        for item in group:
            if item.get("proxy_score") is None:
                item["pareto_kept"] = False
                item["proxy_selected"] = False
                item["candidate_status"] = "invalid"
            elif item not in valid:
                item["proxy_rank"] = None
                item["pareto_kept"] = False
                item["proxy_selection_reasons"] = reasons.get(
                    item["candidate_id"], []
                )
                item["proxy_selected"] = not proxy_screen or item in selected
                item["candidate_status"] = (
                    "proxy_selected_high_risk"
                    if item["proxy_selected"]
                    else "not_run_high_risk"
                )
    for candidate in candidates:
        if candidate.get("proxy_score") is None:
            candidate["proxy_selected"] = False
            candidate["candidate_status"] = "invalid"


def _load_reuse_index(
    results_dir: Path | None,
    fingerprints_by_case: dict[str, dict[str, str | None]],
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    if results_dir is None:
        return {}
    summary_path = results_dir / "phase4_matrix.json"
    if not summary_path.is_file():
        raise ValueError("reuse results directory lacks phase4_matrix.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in summary.get("comparisons", []):
        case = row.get("case")
        if case not in fingerprints_by_case or row.get("status") not in {
            "success", "equivalent_plan"
        }:
            continue
        if any(
            row.get(field) != fingerprints_by_case[case].get(field)
            for field in (
                "graph_hash", "config_hash", "official_code_hash", "solver_code_hash"
            )
        ):
            continue
        raw_value = row.get("raw_result_path")
        raw_path = Path(raw_value) if raw_value else None
        if raw_path is not None and not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        if (
            raw_path is None
            or not raw_path.is_file()
            or not row.get("raw_result_hash")
            or _hash_file(raw_path) != row["raw_result_hash"]
        ):
            continue
        try:
            raw_result = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if raw_result.get("makespan") != row.get("makespan"):
            continue
        rows[(case, row["ncores"], row["evaluation_problem"], row["plan_hash"])] = {
            **row,
            "status": "success",
        }
    return rows


def _reserve_statistics(
    candidates: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    reserved_ids = {
        candidate["candidate_id"]
        for candidate in candidates
        if candidate.get("candidate_status") == "proxy_selected_high_risk"
    }
    evaluated = {
        row["candidate_id"]: row
        for row in comparisons
        if row.get("candidate_id") in reserved_ids
        and row.get("evaluation_problem") == 2
        and row.get("status") in {"success", "equivalent_plan"}
    }
    baselines: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in comparisons:
        if (
            str(row.get("arm", "")).startswith("contiguous_p")
            and row.get("evaluation_problem") == 2
            and row.get("status") in {"success", "equivalent_plan"}
        ):
            baselines.setdefault((row["case"], row["ncores"]), []).append(row)
    wins = ties = losses = 0
    outcomes = []
    for candidate_id, row in evaluated.items():
        baseline_rows = baselines.get((row["case"], row["ncores"]), [])
        baseline = min(
            baseline_rows,
            key=lambda item: (
                item.get("makespan", float("inf")),
                item.get("added_copy_bytes", float("inf")),
            ),
            default=None,
        )
        if baseline is None:
            continue
        candidate_objective = (
            row.get("makespan", float("inf")),
            row.get("added_copy_bytes", float("inf")),
        )
        baseline_objective = (
            baseline.get("makespan", float("inf")),
            baseline.get("added_copy_bytes", float("inf")),
        )
        outcome = (
            "win" if candidate_objective < baseline_objective
            else "tie" if candidate_objective == baseline_objective
            else "loss"
        )
        outcomes.append({
            "candidate_id": candidate_id,
            "case": row["case"],
            "ncores": row["ncores"],
            "candidate_makespan": row.get("makespan"),
            "baseline_makespan": baseline.get("makespan"),
            "outcome": outcome,
        })
        if outcome == "win":
            wins += 1
        elif outcome == "tie":
            ties += 1
        else:
            losses += 1
    evaluated_count = len(evaluated)
    return {
        "high_risk_reserved_count": len(reserved_ids),
        "high_risk_evaluated_count": evaluated_count,
        "high_risk_not_run_count": len(reserved_ids) - evaluated_count,
        "high_risk_wins": wins,
        "high_risk_ties": ties,
        "high_risk_losses": losses,
        "high_risk_recovery_rate": wins / evaluated_count if evaluated_count else None,
        "high_risk_outcomes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 stratified proxy-screened matrix."
    )
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--ncores", nargs="+", type=int, default=[2, 4, 5])
    parser.add_argument("--group-counts", nargs="+", type=int, default=GROUP_COUNTS)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument(
        "--evaluation-problems", nargs="+", type=int,
        choices=(1, 2, 3), default=[2, 3],
    )
    parser.add_argument("--schedule-problem", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--placement-scoring", choices=("baseline", "soft"),
        default=None,
    )
    parser.add_argument(
        "--cache-ordering", choices=("fifo", "reuse_distance"),
        default="fifo",
    )
    parser.add_argument("--no-proxy-screen", action="store_true")
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--max-proxy-candidates", type=int, default=3)
    parser.add_argument("--boundary-ratio-limit", type=float, default=2.0)
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
    if not args.ncores or len(set(args.ncores)) != len(args.ncores):
        parser.error("ncores must be non-empty and unique")
    if any(core_count not in (1, 2, 3, 4, 5) for core_count in args.ncores):
        parser.error("ncores must be selected from 1, 2, 3, 4, 5")
    if not args.group_counts or len(set(args.group_counts)) != len(args.group_counts):
        parser.error("group-counts must be non-empty and unique")
    if any(group_count < 1 for group_count in args.group_counts):
        parser.error("group-counts must be positive")
    if not args.evaluation_problems or len(set(args.evaluation_problems)) != len(args.evaluation_problems):
        parser.error("evaluation-problems must be non-empty and unique")
    if args.evaluator_timeout <= 0:
        parser.error("evaluator-timeout must be positive")
    if args.max_proxy_candidates < 1 or args.boundary_ratio_limit <= 0:
        parser.error("proxy candidate limit and boundary ratio limit must be positive")

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
        else ROOT / "results_round4" / EXPERIMENT_ID / run_id
    )
    if results_dir.exists() and any(results_dir.iterdir()):
        parser.error("results directory is not empty; use a new run-id or directory")

    fingerprints = {
        case: build_fingerprints(
            path, config["path"], official_root, ROOT / "solver"
        )
        for case, path in graph_paths.items()
    }
    matrix_spec = {
        "cases": args.cases,
        "ncores": args.ncores,
        "group_counts": args.group_counts,
        "arms": args.arms,
        "evaluation_problems": args.evaluation_problems,
        "schedule_problem": args.schedule_problem,
        "cache_ordering": args.cache_ordering,
        "placement_scoring_override": args.placement_scoring,
        "max_proxy_candidates": args.max_proxy_candidates,
        "boundary_ratio_limit": args.boundary_ratio_limit,
        "proxy_screen": not args.no_proxy_screen,
    }
    matrix_spec_hash = _matrix_hash(matrix_spec)
    candidates = _candidate_matrix(
        args.cases, args.ncores, args.group_counts, args.arms, run_id
    )
    for candidate in candidates:
        candidate["cache_ordering"] = args.cache_ordering
        if args.placement_scoring is not None:
            candidate["placement_scoring"] = args.placement_scoring
    planned = [
        {**candidate, "evaluation_problem": problem, "status": "not_run"}
        for candidate in candidates
        for problem in args.evaluation_problems
    ]
    run_metadata = make_run_metadata(
        EXPERIMENT_ID,
        run_id,
        next(iter(fingerprints.values())),
        {
            "matrix_spec": matrix_spec,
            "matrix_spec_hash": matrix_spec_hash,
            "evaluator_timeout_sec": args.evaluator_timeout,
            "retain_traces": args.retain_traces,
            "proxy_only": args.proxy_only,
            "official_trace_output_generated": True,
        },
    )
    try:
        ensure_run_metadata(results_dir / "run_metadata.json", run_metadata)
    except ValueError as error:
        parser.error(str(error))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "matrix_spec": matrix_spec,
        "matrix_spec_hash": matrix_spec_hash,
        "planned_candidate_count": len(candidates),
        "planned_evaluation_count": len(planned),
        "planned": planned,
        "status": "generating",
    }
    _write_json(results_dir / "batch_manifest.json", manifest)

    graphs = {case: load_graph(path) for case, path in graph_paths.items()}
    analyses = {case: analyze_graph(graph) for case, graph in graphs.items()}
    evaluator = Evaluator(
        official_root,
        config["path"],
        results_dir,
        args.evaluator_timeout,
        retain_traces=args.retain_traces,
    )
    reuse_rows = _load_reuse_index(
        args.reuse_results_dir.resolve() if args.reuse_results_dir else None,
        fingerprints,
    )
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        graph = graphs[candidate["case"]]
        analysis = analyses[candidate["case"]]
        started = time.monotonic()
        placement_diagnostics: dict[str, Any] = {}
        try:
            partition = partition_with_strategy(
                analysis,
                candidate["group_count"],
                candidate["partition_problem"],
                partition_strategy=candidate["partition_strategy"],
            )
            plan = schedule_partition(
                analysis,
                partition,
                candidate["ncores"],
                problem=args.schedule_problem,
                config=config,
                diagnostics=placement_diagnostics,
                placement_scoring=candidate["placement_scoring"],
                cache_ordering=args.cache_ordering,
            )
            validate_plan(graph, plan, candidate["ncores"])
            plan_hash = _hash_plan(plan)
            metrics = _plan_schedule_metrics(graph, plan, analysis)
            plan_path = results_dir / "schedules" / f"{candidate['candidate_id']}.json"
            _write_json(plan_path, plan)
            candidate.update({
                "plan_hash": plan_hash,
                "plan_path": _portable_path(plan_path),
                "actual_group_count": len(partition.groups),
                "plan_generation_wall_time_sec": time.monotonic() - started,
                "cache_accesses": placement_diagnostics.get("cache_accesses", 0),
                "reuse_distance_bytes": placement_diagnostics.get(
                    "reuse_distance_bytes", 0
                ),
                "reuse_distance_bytes_by_tensor": placement_diagnostics.get(
                    "reuse_distance_bytes_by_tensor", {}
                ),
                "boundary_tensor_count": placement_diagnostics.get(
                    "boundary_tensor_count", 0
                ),
                "boundary_tensor_bytes": placement_diagnostics.get(
                    "boundary_tensor_bytes", 0
                ),
                "boundary_direct_edge_bytes": placement_diagnostics.get(
                    "boundary_direct_edge_bytes", 0
                ),
                "boundary_bytes_total": placement_diagnostics.get(
                    "boundary_bytes_total", 0
                ),
                "boundary_copy_bytes_est": placement_diagnostics.get(
                    "boundary_copy_bytes_est", 0
                ),
                "core_l1_peak_est": placement_diagnostics.get(
                    "core_l1_peak_est", {}
                ),
                "core_ub_peak_est": placement_diagnostics.get(
                    "core_ub_peak_est", {}
                ),
                "max_l1_overflow_bytes_est": placement_diagnostics.get(
                    "max_l1_overflow_bytes_est", 0
                ),
                "max_ub_overflow_bytes_est": placement_diagnostics.get(
                    "max_ub_overflow_bytes_est", 0
                ),
                "l1_residence_bytes_est": placement_diagnostics.get(
                    "l1_residence_bytes_est", {}
                ),
                "ub_residence_bytes_est": placement_diagnostics.get(
                    "ub_residence_bytes_est", {}
                ),
                "long_lived_tensor_bytes": placement_diagnostics.get(
                    "long_lived_tensor_bytes", 0
                ),
                **metrics,
                "proxy_score": _proxy_score(metrics, placement_diagnostics),
                "generation_status": "generated",
                **fingerprints[candidate["case"]],
            })
            diagnostics.append({
                **candidate,
                "placement_diagnostics": placement_diagnostics,
            })
            candidate["_plan"] = plan
        except (ValueError, KeyError, RuntimeError, TypeError) as error:
            candidate.update({
                "plan_hash": None,
                "plan_path": None,
                "plan_generation_wall_time_sec": time.monotonic() - started,
                "proxy_score": None,
                "generation_status": "invalid",
                "error": str(error),
                **fingerprints[candidate["case"]],
            })
            diagnostics.append(dict(candidate))

    _rank_proxy_candidates(
        candidates,
        not args.no_proxy_screen,
        max_selected=args.max_proxy_candidates,
        boundary_ratio_limit=args.boundary_ratio_limit,
    )
    diagnostics_by_id = {
        item["candidate_id"]: item for item in diagnostics
    }
    for candidate in candidates:
        diagnostics_by_id[candidate["candidate_id"]].update({
            key: candidate.get(key)
            for key in (
                "proxy_rank",
                "proxy_selected",
                "candidate_status",
                "safety_status",
                "gate_decision",
                "gate_reason",
                "boundary_ratio_to_contiguous",
                "overflow_delta_to_contiguous",
                "boundary_copy_delta_to_contiguous",
                "l1_overflow_delta_to_contiguous",
                "ub_overflow_delta_to_contiguous",
                "max_core_compute_delta_to_contiguous",
                "m_load_imbalance_delta_to_contiguous",
                "v_load_imbalance_delta_to_contiguous",
                "active_core_delta_to_contiguous",
                "critical_path_cross_core_delta_to_contiguous",
                "residence_reduction_per_extra_copy_byte",
                "overflow_to_boundary_ratio",
                "pareto_kept",
                "proxy_selection_reasons",
            )
        })
    _write_json(results_dir / "subgraph_diagnostics.json", {
        "matrix_spec_hash": matrix_spec_hash,
        "candidates": diagnostics,
    })
    manifest["status"] = "proxy_screened"
    manifest["reserve_statistics"] = _reserve_statistics(candidates, [])
    manifest["proxy_selected_count"] = sum(
        candidate.get("proxy_selected", False) for candidate in candidates
    )
    manifest["proxy_invalid_count"] = sum(
        candidate.get("candidate_status") == "invalid" for candidate in candidates
    )
    _write_json(results_dir / "batch_manifest.json", manifest)

    comparisons: list[dict[str, Any]] = []
    evaluation_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    ledger_path = results_dir / "candidate_trials.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")

    def persist() -> None:
        by_key = {
            (row["candidate_id"], row["evaluation_problem"]): row
            for row in comparisons
        }
        for item in planned:
            outcome = by_key.get((item["candidate_id"], item["evaluation_problem"]))
            if outcome is not None:
                for field in (
                    "status", "makespan", "plan_hash", "evaluation_reused",
                    "raw_result_path", "error",
                ):
                    if field in outcome:
                        item[field] = outcome[field]
        manifest["completed_evaluation_count"] = len(comparisons)
        manifest["official_evaluation_calls"] = evaluator.calls
        manifest["reserve_statistics"] = _reserve_statistics(candidates, comparisons)
        manifest["status_counts"] = {
            status: sum(row.get("status") == status for row in comparisons)
            for status in sorted({row.get("status") for row in comparisons})
        }
        _write_json(results_dir / "batch_manifest.json", manifest)
        _write_json(results_dir / "phase4_matrix.json", {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "matrix_spec": matrix_spec,
            "matrix_spec_hash": matrix_spec_hash,
            "comparisons": comparisons,
            "reserve_statistics": manifest["reserve_statistics"],
        })

    total = len(planned)
    done = 0
    for candidate in candidates:
        for problem in args.evaluation_problems:
            done += 1
            record = {
                key: value for key, value in candidate.items() if not key.startswith("_")
            }
            record["evaluation_problem"] = problem
            record["evaluation_reused"] = False
            if candidate.get("generation_status") != "generated":
                record.update({
                    "status": "invalid",
                    "error": candidate.get("error"),
                })
            elif not candidate.get("proxy_selected", False):
                record["status"] = candidate.get("candidate_status", "screened_out")
            elif args.proxy_only:
                record["status"] = "proxy_selected"
            else:
                key = (
                    candidate["case"], candidate["ncores"],
                    problem, candidate["plan_hash"],
                )
                cached = evaluation_cache.get(key) or reuse_rows.get(key)
                if cached is not None:
                    record.update({
                        field: cached.get(field) for field in RESULT_FIELDS
                    })
                    record.update({
                        "status": "equivalent_plan",
                        "evaluation_reused": True,
                        "evaluation_source_candidate_id": cached.get(
                            "candidate_id"
                        ),
                    })
                else:
                    tag = f"{candidate['candidate_id']}_evalp{problem}"
                    try:
                        evaluated = evaluator.evaluate_multicore(
                            graphs[candidate["case"]].path,
                            candidate["_plan"],
                            problem,
                            tag,
                        )
                        result = evaluated.result
                        movement = result.get("data_movement_bytes", {})
                        cache_stats = result.get("cache_stats", {})
                        raw_path = results_dir / "raw" / f"{Evaluator._safe_tag(tag)}.json"
                        record.update({
                            "status": "success" if result.get("makespan") is not None else "invalid",
                            "makespan": result.get("makespan"),
                            "added_copy_bytes": movement.get("added_copy_bytes"),
                            "partition_added_copy_bytes": movement.get("partition_added_copy_bytes"),
                            "spill_added_copy_bytes": movement.get("spill_added_copy_bytes"),
                            "cache_hit_bytes": cache_stats.get("hit_bytes"),
                            "cache_miss_bytes": cache_stats.get("miss_bytes"),
                            "cache_hit_rate": cache_stats.get("hit_rate"),
                            "cache_hits": cache_stats.get("hits"),
                            "cache_accesses": cache_stats.get("accesses"),
                            "memory_peak_by_core": result.get("memory_peak_by_core"),
                            "raw_result_path": _portable_path(raw_path),
                            "raw_result_hash": _hash_file(raw_path) if raw_path.is_file() else None,
                            "evaluation_wall_time_sec": evaluated.elapsed_sec,
                            "error": None,
                            "evaluation_source_candidate_id": candidate["candidate_id"],
                        })
                    except EvaluationError as error:
                        record.update({
                            "status": "timeout" if "timeout" in str(error).lower() else "evaluation_failed",
                            "evaluation_wall_time_sec": error.elapsed_sec,
                            "error": str(error),
                            "evaluation_source_candidate_id": candidate["candidate_id"],
                        })
                    evaluation_cache[key] = {
                        field: record.get(field) for field in RESULT_FIELDS
                    }
                    evaluation_cache[key]["status"] = record["status"]
                    evaluation_cache[key]["candidate_id"] = candidate["candidate_id"]

            comparisons.append(record)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            persist()
            print(
                f"[{done}/{total}] {candidate['case']} n{candidate['ncores']} "
                f"{candidate['arm']} g{candidate['group_count']} p{problem} "
                f"{record['status']}",
                flush=True,
            )

    manifest["status"] = "completed"
    manifest["completed_evaluation_count"] = len(comparisons)
    manifest["official_evaluation_calls"] = evaluator.calls
    _write_json(results_dir / "batch_manifest.json", manifest)
    _write_json(results_dir / "run_metadata.json", run_metadata | {
        "completed_evaluation_count": len(comparisons),
        "official_evaluation_calls": evaluator.calls,
        "reserve_statistics": _reserve_statistics(candidates, comparisons),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())