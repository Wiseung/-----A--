from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from . import cache_aware, schedule_a, schedule_b
from .evaluate_adapter import EvaluationError, Evaluator
from .graph_analysis import GraphAnalysis, analyze_graph
from .graph_io import Graph, load_config, load_graph
from .improve import generate_neighbor_candidates, plan_signature
from .legality import validate_plan
from .run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)


SUMMARY_FIELDS = [
    "experiment_id", "run_id", "case", "problem", "ncores", "method",
    "graph_hash", "config_hash", "official_code_hash", "solver_code_hash",
    "time_limit_sec", "baseline_timeout_sec", "evaluator_timeout_sec",
    "max_evals", "improve_enabled", "retry_baseline", "placement_scoring",
    "cache_ordering", "candidate_selection", "residence_ordering",
    "proxy_max_candidates",
    "proxy_boundary_ratio_limit", "candidate_id",
    "proxy_score", "proxy_rank", "pareto_kept", "proxy_selected",
    "candidate_status", "proxy_selection_reasons", "safety_status",
    "best_candidate_source", "makespan",
    "single_core_baseline", "speedup", "added_copy_bytes",
    "original_copy_bytes", "scheduled_copy_bytes", "cache_hit_rate",
    "subgraph_count", "solver_runtime_sec", "evaluator_runtime_sec",
    "baseline_cache_hit",
    "baseline_evaluator_runtime_sec", "baseline_wall_sec", "search_wall_sec",
    "candidate_eval_wall_sec", "solver_overhead_sec", "total_wall_sec",
    "evaluator_calls", "seed",
    "makespan_no_l2", "makespan_l2", "cache_speedup", "status", "error",
]

CandidateSpec = tuple[
    dict[str, Any], str, dict[str, Any] | None, str | None
]


def _read_singlecore(
    results_dir: Path,
    graph: Graph,
    run_id: str,
    fingerprints: dict[str, str | None],
) -> dict[str, Any] | None:
    path = results_dir / "raw" / f"{graph.path.stem}_singlecore_{run_id}.json"
    metadata_path = path.with_suffix(".metadata.json")
    if path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            baseline = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if all(metadata.get(key) == fingerprints.get(key) for key in (
            "graph_hash", "config_hash", "official_code_hash"
        )):
            return baseline
    return None


def _singlecore_failure(results_dir: Path, graph: Graph, run_id: str) -> Path:
    return results_dir / "logs" / f"{graph.path.stem}_singlecore_{run_id}.failed.txt"


def _save_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_saved_plan(
    path: Path,
    fingerprints: dict[str, str | None],
    expected: dict[str, Any],
    allow_unverified: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        if not allow_unverified:
            return None
        metadata = None
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if any(metadata.get(key) != value for key, value in {
            **fingerprints,
            **expected,
        }.items()):
            return None
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata is not None and metadata.get("plan_hash") != hashlib.sha256(
        plan_signature(plan).encode("utf-8")
    ).hexdigest():
        return None
    return plan if isinstance(plan, dict) else None


def _saved_plan_paths(
    directory: Path, case: str, problem: int, ncores: int
) -> list[Path]:
    schedule_dir = directory / "schedules"
    legacy = schedule_dir / f"{case}_p{problem}_n{ncores}_best.json"
    validated = sorted(schedule_dir.glob(
        f"{case}_p{problem}_n{ncores}_*_best_validated.json"
    ))
    return [legacy, *validated]


def _store_summary(results_dir: Path, row: dict[str, Any]) -> None:
    json_path = results_dir / "summary.json"
    csv_path = results_dir / "summary.csv"
    previous = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else []
    identity_fields = (
        "experiment_id", "run_id", "case", "problem", "ncores",
        "graph_hash", "config_hash", "official_code_hash", "solver_code_hash",
        "method", "seed", "time_limit_sec", "baseline_timeout_sec",
        "evaluator_timeout_sec", "max_evals", "improve_enabled",
        "retry_baseline", "placement_scoring", "cache_ordering",
        "candidate_selection", "residence_ordering", "proxy_max_candidates",
        "proxy_boundary_ratio_limit",
    )
    key = tuple(row.get(field) for field in identity_fields)
    previous = [
        item for item in previous
        if tuple(item.get(field) for field in identity_fields) != key
    ]
    previous.append(row)
    previous.sort(key=lambda item: (
        item.get("experiment_id") or "", item.get("run_id") or "",
        item["case"], item["problem"], item["ncores"],
    ))
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(previous, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_path)
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(previous)
    csv_temporary.replace(csv_path)


def _append_candidate_trial(results_dir: Path, record: dict[str, Any]) -> None:
    path = results_dir / "candidate_trials.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _append_subgraph_diagnostic(results_dir: Path, record: dict[str, Any]) -> None:
    path = results_dir / "subgraph_diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _portable_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_load_imbalance(loads: list[int]) -> float:
    total = sum(loads)
    if not loads or total == 0:
        return 0.0
    mean = total / len(loads)
    return max(loads) / mean - 1.0


def _plan_schedule_metrics(
    graph: Graph,
    plan: dict[str, Any],
    analysis: GraphAnalysis | None = None,
) -> dict[str, Any]:
    analysis = analysis or analyze_graph(graph)
    owners = {
        group_id: core_id
        for core_id, schedule in enumerate(plan["core_schedules"])
        for group_id in schedule
    }
    groups_by_core = {
        str(core_id): list(schedule)
        for core_id, schedule in enumerate(plan["core_schedules"])
    }
    pipe_cycles: dict[int, dict[str, int]] = {}
    for core_id in range(len(plan["core_schedules"])):
        pipe_cycles[core_id] = {"PIPE_M": 0, "PIPE_V": 0}
    node_to_subgraph = {
        int(op_key): group_id
        for op_key, group_id in plan["node_to_subgraph"].items()
    }
    for op_id, group_id in node_to_subgraph.items():
        op = graph.ops[op_id]
        if op["pipe"] in {"PIPE_M", "PIPE_V"}:
            core_id = owners[group_id]
            pipe_cycles[core_id][op["pipe"]] += max(1, op["cycles"])
    m_cycles = [pipe_cycles[core_id]["PIPE_M"]
                for core_id in range(len(plan["core_schedules"]))]
    v_cycles = [pipe_cycles[core_id]["PIPE_V"]
                for core_id in range(len(plan["core_schedules"]))]
    critical_path_length = max(analysis.forward_path.values(), default=0)
    critical_ops = {
        op_id for op_id in analysis.eligible_ops
        if (
            analysis.forward_path[op_id]
            + analysis.backward_path[op_id]
            - max(1, graph.ops[op_id]["cycles"])
            == critical_path_length
        )
    }
    critical_groups: set[int] = set()
    for op_id in critical_ops:
        critical_groups.add(node_to_subgraph[op_id])
    critical_group_count_by_core = {
        str(core_id): sum(
            group_id in critical_groups
            for group_id in schedule
        )
        for core_id, schedule in enumerate(plan["core_schedules"])
    }
    owners_by_group = owners
    critical_path_cross_core_edges = 0
    for source in analysis.eligible_ops:
        if source not in node_to_subgraph:
            continue
        if source not in critical_ops:
            continue
        source_core = owners_by_group[node_to_subgraph[source]]
        for target in analysis.successors[source]:
            if target in critical_ops and source_core != owners_by_group[
                node_to_subgraph[target]
            ]:
                critical_path_cross_core_edges += 1
    core_compute = [max(m, v) for m, v in zip(m_cycles, v_cycles)]
    return {
        "active_core_count": sum(bool(schedule) for schedule in plan["core_schedules"]),
        "groups_by_core": groups_by_core,
        "m_cycles_by_core": {
            str(core_id): values["PIPE_M"] for core_id, values in pipe_cycles.items()
        },
        "v_cycles_by_core": {
            str(core_id): values["PIPE_V"] for core_id, values in pipe_cycles.items()
        },
        "max_core_compute": max(core_compute, default=0),
        "m_load_imbalance": _relative_load_imbalance(m_cycles),
        "v_load_imbalance": _relative_load_imbalance(v_cycles),
        "critical_group_count_by_core": critical_group_count_by_core,
        "critical_path_cross_core_edges": critical_path_cross_core_edges,
    }


def _objective(result: dict[str, Any]) -> tuple[int, int]:
    movement = result.get("data_movement_bytes", {})
    return result["makespan"], movement.get("added_copy_bytes", 0)


def _generator(problem: int) -> tuple[Callable[..., dict[str, Any]], Callable[..., list[int]], str]:
    if problem == 1:
        return schedule_a.generate_schedule, schedule_a.candidate_counts, "tensor_cut_critical_list"
    if problem == 2:
        return schedule_b.generate_schedule, schedule_b.candidate_counts, "co_location_critical_list"
    return cache_aware.generate_schedule, cache_aware.candidate_counts, "fifo_cache_aware_list"


def _proxy_score(
    metrics: dict[str, Any],
    placement_diagnostics: dict[str, Any] | None,
) -> tuple[float, float, float, float, float, float, float, int, int]:
    diagnostics = placement_diagnostics or {}
    estimated_finish = max(
        (
            float(item.get("group_finish", 0.0))
            for item in diagnostics.get("groups", [])
        ),
        default=0.0,
    )
    added_copy = diagnostics.get(
        "estimated_partition_added_copy_bytes_recomputed"
    )
    if added_copy is None:
        added_copy = diagnostics.get(
            "estimated_partition_added_copy_bytes", 10**30
        )
    imbalance = max(
        float(metrics.get("m_load_imbalance", 0.0)),
        float(metrics.get("v_load_imbalance", 0.0)),
    )
    boundary_bytes = diagnostics.get(
        "boundary_copy_bytes_est",
        diagnostics.get(
            "boundary_bytes_total",
            diagnostics.get("boundary_tensor_bytes", added_copy),
        ),
    )
    overflow_bytes = (
        int(diagnostics.get("max_l1_overflow_bytes_est", 0))
        + int(diagnostics.get("max_ub_overflow_bytes_est", 0))
    )
    l1_residence = diagnostics.get("l1_residence_bytes_est", {})
    ub_residence = diagnostics.get("ub_residence_bytes_est", {})
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
        int(diagnostics.get("estimated_repeated_input_bytes", 0)),
        -int(metrics.get("active_core_count", 0)),
    )


def _proxy_objective(
    score: tuple[float, float, float, float, float, float, float, int, int],
) -> tuple[float, float, float, float, float, float]:
    return (
        float(score[0]),
        float(score[1]),
        float(score[2]),
        float(score[4]),
        float(score[5]),
        float(score[6]),
    )


def _select_proxy_candidates(
    entries: list[dict[str, Any]],
    evaluation_limit: int,
    max_selected: int,
    boundary_ratio_limit: float,
) -> set[int]:
    for entry in entries:
        metrics = entry.get("metrics")
        entry["proxy_score"] = (
            _proxy_score(metrics, entry.get("placement_diagnostics"))
            if metrics is not None else None
        )

    valid = [entry for entry in entries if entry.get("proxy_score") is not None]
    if not valid:
        for entry in entries:
            entry["selection_metadata"] = {
                "proxy_score": None,
                "proxy_rank": None,
                "pareto_kept": False,
                "proxy_selected": False,
                "candidate_status": "invalid",
                "proxy_selection_reasons": [],
                "safety_status": "invalid",
            }
        return set()

    incumbent = valid[0]
    baseline_score = incumbent["proxy_score"]
    baseline_diagnostics = incumbent.get("placement_diagnostics") or {}
    baseline_boundary = float(
        baseline_diagnostics.get(
            "boundary_copy_bytes_est",
            baseline_diagnostics.get(
                "boundary_bytes_total",
                baseline_diagnostics.get(
                    "boundary_tensor_bytes", baseline_score[1]
                ),
            ),
        )
    )
    baseline_overflow = int(
        baseline_diagnostics.get("max_l1_overflow_bytes_est", 0)
    ) + int(baseline_diagnostics.get("max_ub_overflow_bytes_est", 0))

    for entry in valid:
        if entry is incumbent:
            entry["safety_status"] = "baseline_or_uncertain"
            entry["boundary_ratio_to_incumbent"] = None
            entry["overflow_delta_to_incumbent"] = None
            continue
        diagnostics = entry.get("placement_diagnostics") or {}
        boundary = float(
            diagnostics.get(
                "boundary_copy_bytes_est",
                diagnostics.get(
                    "boundary_bytes_total",
                    diagnostics.get("boundary_tensor_bytes", entry["proxy_score"][1]),
                ),
            )
        )
        overflow = int(diagnostics.get("max_l1_overflow_bytes_est", 0)) + int(
            diagnostics.get("max_ub_overflow_bytes_est", 0)
        )
        ratio = (
            boundary / baseline_boundary
            if baseline_boundary > 0
            else (float("inf") if boundary > 0 else 1.0)
        )
        entry["boundary_ratio_to_incumbent"] = ratio
        entry["overflow_delta_to_incumbent"] = overflow - baseline_overflow
        entry["safety_status"] = (
            "high_risk"
            if ratio > boundary_ratio_limit and overflow >= baseline_overflow
            else "safe_or_uncertain"
        )

    safe = [
        entry for entry in valid
        if entry.get("safety_status") != "high_risk"
    ]

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_objective = _proxy_objective(left["proxy_score"])
        right_objective = _proxy_objective(right["proxy_score"])
        return (
            all(a <= b for a, b in zip(left_objective, right_objective))
            and any(a < b for a, b in zip(left_objective, right_objective))
        )

    pareto = [
        entry for entry in safe
        if not any(
            other is not entry and dominates(other, entry)
            for other in safe
        )
    ]
    risk_candidates = [
        entry for entry in valid
        if entry.get("safety_status") == "high_risk"
    ]
    risk_candidate = min(
        risk_candidates,
        key=lambda entry: (tuple(entry["proxy_score"]), entry["index"]),
        default=None,
    )
    evaluation_limit = max(1, evaluation_limit)
    max_selected = max(1, max_selected)
    risk_slot = risk_candidate is not None and evaluation_limit > 1
    safe_limit = min(max_selected, evaluation_limit - int(risk_slot))
    selected: list[dict[str, Any]] = [incumbent]
    reasons: dict[int, list[str]] = {incumbent["index"]: ["incumbent"]}
    criterion_order = (
        (0, "estimated_finish"),
        (1, "boundary_copy"),
        (2, "overflow"),
        (3, "m_v_balance"),
    )
    selection_pool = pareto or safe
    for criterion, reason in criterion_order:
        if len(selected) >= safe_limit:
            break
        candidate = min(
            selection_pool,
            key=lambda entry: (
                _proxy_objective(entry["proxy_score"])[criterion],
                entry["index"],
            ),
            default=None,
        )
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
            reasons.setdefault(candidate["index"], []).append(reason)
    for candidate in sorted(
        pareto,
        key=lambda entry: (tuple(entry["proxy_score"]), entry["index"]),
    ):
        if len(selected) >= safe_limit:
            break
        if candidate not in selected:
            selected.append(candidate)
            reasons.setdefault(candidate["index"], []).append("pareto")
    if risk_candidate is not None and len(selected) < evaluation_limit:
        selected.append(risk_candidate)
        reasons.setdefault(risk_candidate["index"], []).append(
            "high_risk_reserve"
        )

    selected_indices = {entry["index"] for entry in selected}
    ranked = sorted(
        valid,
        key=lambda entry: (tuple(entry["proxy_score"]), entry["index"]),
    )
    ranks = {entry["index"]: rank for rank, entry in enumerate(ranked, start=1)}
    for entry in entries:
        index = entry["index"]
        score = entry.get("proxy_score")
        if score is None:
            entry["selection_metadata"] = {
                "proxy_score": None,
                "proxy_rank": None,
                "pareto_kept": False,
                "proxy_selected": False,
                "candidate_status": "invalid",
                "proxy_selection_reasons": [],
                "safety_status": "invalid",
            }
            continue
        selected_entry = index in selected_indices
        safety_status = entry.get("safety_status", "safe_or_uncertain")
        entry["selection_metadata"] = {
            "proxy_score": score,
            "proxy_rank": ranks[index],
            "pareto_kept": entry in pareto,
            "proxy_selected": selected_entry,
            "candidate_status": (
                "proxy_selected_high_risk"
                if selected_entry and safety_status == "high_risk"
                else "proxy_selected"
                if selected_entry
                else "not_run_high_risk"
                if safety_status == "high_risk"
                else "screened_out"
            ),
            "proxy_selection_reasons": reasons.get(index, []),
            "safety_status": safety_status,
            "boundary_ratio_to_incumbent": entry.get(
                "boundary_ratio_to_incumbent"
            ),
            "overflow_delta_to_incumbent": entry.get(
                "overflow_delta_to_incumbent"
            ),
        }
    return selected_indices


def solve_one(
    graph: Graph,
    problem: int,
    ncores: int,
    config: dict[str, Any],
    evaluator: Evaluator,
    results_dir: Path,
    time_limit: float,
    max_evals: int,
    seed: int,
    improve: bool,
    baseline_timeout: float,
    retry_baseline: bool,
    experiment_id: str = "round2",
    run_id: str | None = None,
    history_dir: Path | None = None,
    placement_scoring: str = "baseline",
    cache_ordering: str = "fifo",
    candidate_selection: str = "ordered",
    residence_ordering: str = "off",
    proxy_max_candidates: int = 4,
    proxy_boundary_ratio_limit: float = 2.0,
) -> dict[str, Any]:
    run_id = run_id or new_run_id()
    if not valid_run_label(experiment_id) or not valid_run_label(run_id):
        raise ValueError("experiment_id and run_id must be path-safe labels")
    if candidate_selection not in {"ordered", "proxy_pareto"}:
        raise ValueError("candidate_selection must be ordered or proxy_pareto")
    if residence_ordering not in {"off", "neighbor"}:
        raise ValueError("residence_ordering must be off or neighbor")
    if proxy_max_candidates < 1 or proxy_boundary_ratio_limit <= 0:
        raise ValueError("proxy selection limits must be positive")
    history_dir = history_dir or Path(__file__).resolve().parents[1] / "results"
    fingerprints = build_fingerprints(
        graph.path,
        getattr(evaluator, "config_path", Path()),
        getattr(
            evaluator,
            "official_root",
            Path(__file__).resolve().parents[1] / "2026_official",
        ),
        Path(__file__).resolve().parent,
    )
    total_started = time.monotonic()
    baseline_started = time.monotonic()
    calls_before = evaluator.calls
    baseline_runtime = 0.0
    errors = []
    per_candidate_timeout = evaluator.timeout_sec

    baseline = _read_singlecore(results_dir, graph, run_id, fingerprints)
    baseline_cache_hit = baseline is not None
    baseline_failure = _singlecore_failure(results_dir, graph, run_id)
    if baseline is None and baseline_failure.is_file() and not retry_baseline:
        errors.append("singlecore baseline previously failed; see " + str(baseline_failure))
    elif baseline is None:
        evaluator.timeout_sec = baseline_timeout
        try:
            single = evaluator.evaluate_singlecore(
                graph.path, f"{graph.path.stem}_singlecore_{run_id}"
            )
            baseline = single.result
            baseline_runtime = single.elapsed_sec
            baseline_metadata_path = (
                results_dir / "raw" /
                f"{graph.path.stem}_singlecore_{run_id}.metadata.json"
            )
            _save_plan(baseline_metadata_path, {
                key: fingerprints[key]
                for key in ("graph_hash", "config_hash", "official_code_hash")
            })
        except EvaluationError as error:
            baseline_runtime += error.elapsed_sec
            errors.append(f"singlecore baseline: {error}")

    baseline_wall_sec = time.monotonic() - baseline_started
    evaluator.timeout_sec = per_candidate_timeout
    search_started = time.monotonic()
    deadline = search_started + time_limit
    baseline_makespan = baseline.get("makespan") if baseline else None
    analysis = analyze_graph(graph)
    generate, get_counts, method = _generator(problem)
    def generate_with_scoring(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["placement_scoring"] = placement_scoring
        kwargs["cache_ordering"] = cache_ordering
        return generate(*args, **kwargs)

    group_counts = get_counts(analysis, ncores)
    run_fields = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        **fingerprints,
        "time_limit_sec": time_limit,
        "baseline_timeout_sec": baseline_timeout,
        "evaluator_timeout_sec": per_candidate_timeout,
        "max_evals": max_evals,
        "improve_enabled": improve,
        "retry_baseline": retry_baseline,
        "placement_scoring": placement_scoring,
        "cache_ordering": cache_ordering,
        "candidate_selection": candidate_selection,
        "residence_ordering": residence_ordering,
        "proxy_max_candidates": proxy_max_candidates,
        "proxy_boundary_ratio_limit": proxy_boundary_ratio_limit,
    }
    best_plan: dict[str, Any] | None = None
    legal_draft: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    best_tag: str | None = None
    best_candidate_id: str | None = None
    best_candidate_source: str | None = None
    best_selection_metadata: dict[str, Any] | None = None
    tested: set[str] = set()
    candidate_sequence = 0
    successful_evals = 0
    attempted_evals = 0
    multicore_eval_runtime = 0.0

    def try_candidate(
        plan: dict[str, Any],
        stage: str,
        placement_diagnostics: dict[str, Any] | None = None,
        source_plan: str | None = None,
        parent_candidate: str | None = None,
        selection_metadata: dict[str, Any] | None = None,
        evaluate_candidate: bool = True,
    ) -> bool:
        nonlocal best_plan, legal_draft, best_result, best_tag, best_candidate_id
        nonlocal best_candidate_source, best_selection_metadata
        nonlocal successful_evals, attempted_evals, multicore_eval_runtime
        nonlocal candidate_sequence
        signature = plan_signature(plan)
        if signature in tested or (
            evaluate_candidate and attempted_evals >= max_evals
        ):
            return False
        tested.add(signature)
        candidate_sequence += 1
        candidate_id = (
            f"{graph.path.stem}_p{problem}_n{ncores}_{run_id}_"
            f"{candidate_sequence:04d}_{uuid.uuid4().hex[:8]}"
        )
        plan_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if stage.startswith("problem2_warm_start"):
            candidate_method = "co_location_critical_list"
        elif stage == "problem2_validated_warm_start":
            candidate_method = "validated_problem2_plan"
        elif stage == "legacy_history_warm_start":
            candidate_method = "legacy_plan"
        elif stage == "validated_history_warm_start":
            candidate_method = "validated_history_plan"
        elif stage == "previous_core_count_warm_start":
            candidate_method = "previous_core_plan_plus_empty_core"
        else:
            candidate_method = method
        effective_selection_metadata = selection_metadata or {
            "proxy_selected": True,
            "candidate_status": "ordered",
            "proxy_selection_reasons": [],
        }
        selection_fields = {
            "candidate_selection": candidate_selection,
            "proxy_max_candidates": proxy_max_candidates,
            "proxy_boundary_ratio_limit": proxy_boundary_ratio_limit,
            **effective_selection_metadata,
        }
        record = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "case": graph.path.stem,
            "problem": problem,
            "ncores": ncores,
            "seed": seed,
            "time_limit_sec": time_limit,
            "baseline_timeout_sec": baseline_timeout,
            "evaluator_timeout_sec": per_candidate_timeout,
            "max_evals": max_evals,
            "improve_enabled": improve,
            "placement_scoring": placement_scoring,
            "residence_ordering": residence_ordering,
            "candidate_family": stage,
            "method": method,
            "candidate_method": candidate_method,
            "partition_strategy": candidate_method,
            "mapping_strategy": candidate_method,
            "ordering_strategy": candidate_method,
            "parent_candidate": parent_candidate,
            "source_plan": _portable_path(Path(source_plan)) if source_plan else None,
            "group_count": len(set(plan.get("node_to_subgraph", {}).values())),
            "plan_hash": plan_hash,
            **selection_fields,
            **fingerprints,
        }
        try:
            validate_plan(graph, plan, ncores)
        except (ValueError, RuntimeError) as error:
            _append_subgraph_diagnostic(results_dir, {
                **record,
                "status": "invalid",
                "placement_estimates": placement_diagnostics,
                "official_makespan": None,
                "accepted": False,
                "rejection_reason": str(error),
            })
            _append_candidate_trial(results_dir, {
                **record,
                "status": "invalid",
                "evaluation_wall_time_sec": 0.0,
                "official_makespan": None,
                "added_copy_bytes": None,
                "accepted": False,
                "rejection_reason": str(error),
            })
            errors.append(f"{candidate_id}: invalid plan: {error}")
            return False
        plan_metrics = _plan_schedule_metrics(graph, plan, analysis)
        _save_plan(results_dir / "schedules" / f"{candidate_id}.json", plan)
        if legal_draft is None:
            legal_draft = plan
            _save_plan(
                results_dir / "schedules" /
                f"{graph.path.stem}_p{problem}_n{ncores}_{run_id}_legal_draft.json",
                legal_draft,
            )
        if not evaluate_candidate:
            rejection_reason = (
                effective_selection_metadata.get("candidate_status", "screened_out")
            )
            _append_subgraph_diagnostic(results_dir, {
                **record,
                **plan_metrics,
                "status": rejection_reason,
                "placement_estimates": placement_diagnostics,
                "official_makespan": None,
                "evaluation_wall_time_sec": 0.0,
                "accepted": False,
                "rejection_reason": rejection_reason,
            })
            _append_candidate_trial(results_dir, {
                **record,
                "status": rejection_reason,
                "evaluation_wall_time_sec": 0.0,
                "official_makespan": None,
                "added_copy_bytes": None,
                "accepted": False,
                "rejection_reason": rejection_reason,
            })
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _append_subgraph_diagnostic(results_dir, {
                **record,
                **plan_metrics,
                "status": "not_run_budget",
                "placement_estimates": placement_diagnostics,
                "official_makespan": None,
                "accepted": False,
                "rejection_reason": "search budget exhausted before evaluation",
            })
            _append_candidate_trial(results_dir, {
                **record,
                "status": "not_run_budget",
                "evaluation_wall_time_sec": 0.0,
                "official_makespan": None,
                "added_copy_bytes": None,
                "accepted": False,
                "rejection_reason": "search budget exhausted before evaluation",
            })
            return False
        tag = candidate_id
        evaluator.timeout_sec = min(per_candidate_timeout, remaining)
        attempted_evals += 1
        try:
            evaluated = evaluator.evaluate_multicore(
                graph.path, plan, problem, tag
            )
        except EvaluationError as error:
            multicore_eval_runtime += error.elapsed_sec
            errors.append(f"{tag}: {error}")
            _append_subgraph_diagnostic(results_dir, {
                **record,
                **plan_metrics,
                "status": "evaluation_failed",
                "placement_estimates": placement_diagnostics,
                "official_makespan": None,
                "evaluation_wall_time_sec": error.elapsed_sec,
                "accepted": False,
                "rejection_reason": str(error),
            })
            _append_candidate_trial(results_dir, {
                **record,
                "status": "evaluation_failed",
                "evaluation_wall_time_sec": error.elapsed_sec,
                "official_makespan": None,
                "added_copy_bytes": None,
                "accepted": False,
                "rejection_reason": str(error),
            })
            return False
        multicore_eval_runtime += evaluated.elapsed_sec
        successful_evals += 1
        accepted = (
            best_result is None
            or _objective(evaluated.result) < _objective(best_result)
        )
        movement = evaluated.result.get("data_movement_bytes", {})
        _append_subgraph_diagnostic(results_dir, {
            **record,
            **plan_metrics,
            "status": "evaluated",
            "placement_estimates": placement_diagnostics,
            "official_makespan": evaluated.result.get("makespan"),
            "official_added_copy_bytes": movement.get("added_copy_bytes"),
            "evaluation_wall_time_sec": evaluated.elapsed_sec,
            "accepted": accepted,
            "rejection_reason": None if accepted else "objective_not_better",
        })
        _append_candidate_trial(results_dir, {
            **record,
            "status": "evaluated",
            "evaluation_wall_time_sec": evaluated.elapsed_sec,
            "official_makespan": evaluated.result.get("makespan"),
            "added_copy_bytes": movement.get("added_copy_bytes"),
            "accepted": accepted,
            "rejection_reason": None if accepted else "objective_not_better",
        })
        if accepted:
            best_plan = plan
            best_result = evaluated.result
            best_tag = tag
            best_candidate_id = candidate_id
            best_candidate_source = stage
            best_selection_metadata = effective_selection_metadata
            validated_path = (
                results_dir / "schedules" /
                f"{graph.path.stem}_p{problem}_n{ncores}_{run_id}_best_validated.json"
            )
            _save_plan(
                validated_path,
                best_plan,
            )
            _save_plan(validated_path.with_suffix(".metadata.json"), {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "problem": problem,
                "ncores": ncores,
                "method": method,
                "seed": seed,
                "time_limit_sec": time_limit,
                "max_evals": max_evals,
                "candidate_id": candidate_id,
                "plan_hash": plan_hash,
                "makespan": evaluated.result.get("makespan"),
                **fingerprints,
            })
        return accepted

    candidate_factories: list[tuple[str, Callable[[], CandidateSpec]]] = []

    def add_plan_factory(
        plan: dict[str, Any],
        stage: str,
        placement_diagnostics: dict[str, Any] | None = None,
        source_plan: str | None = None,
    ) -> None:
        def factory(
            saved_plan: dict[str, Any] = plan,
            candidate_stage: str = stage,
            saved_diagnostics: dict[str, Any] | None = placement_diagnostics,
            saved_source_plan: str | None = source_plan,
        ) -> CandidateSpec:
            return (
                saved_plan,
                candidate_stage,
                saved_diagnostics,
                saved_source_plan,
            )

        candidate_factories.append((stage, factory))

    def add_generated_factory(
        stage: str,
        generator_function: Callable[..., dict[str, Any]],
        group_count: int,
        topology_strategy: str | None = None,
    ) -> None:
        def factory() -> CandidateSpec:
            placement_diagnostics: dict[str, Any] = {}
            generator_kwargs: dict[str, Any] = {
                "diagnostics": placement_diagnostics,
            }
            if topology_strategy is not None:
                generator_kwargs["topology_strategy"] = topology_strategy
            plan = generator_function(
                analysis, ncores, config, group_count, **generator_kwargs
            )
            return plan, stage, placement_diagnostics, None

        candidate_factories.append((stage, factory))

    def generate_problem2_with_scoring(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["placement_scoring"] = placement_scoring
        kwargs["cache_ordering"] = cache_ordering
        return schedule_b.generate_schedule(*args, **kwargs)

    def load_saved_candidate(
        source_problem: int,
        source_ncores: int,
        append_empty_core: bool = False,
        validated_only: bool = False,
    ) -> CandidateSpec | None:
        directories = list(dict.fromkeys((results_dir, history_dir)))
        for directory in directories:
            paths = _saved_plan_paths(
                directory, graph.path.stem, source_problem, source_ncores
            )
            if validated_only:
                paths = paths[1:]
            for path in paths:
                plan = _load_saved_plan(
                    path,
                    fingerprints,
                    {"problem": source_problem, "ncores": source_ncores},
                    allow_unverified=path.name.endswith("_best.json"),
                )
                if plan is None:
                    continue
                if append_empty_core:
                    schedules = plan.get("core_schedules")
                    if not isinstance(schedules, list) or len(schedules) != source_ncores:
                        continue
                    plan = {
                        "node_to_subgraph": plan.get("node_to_subgraph", {}),
                        "core_schedules": [list(schedule) for schedule in schedules] + [[]],
                    }
                if path.with_suffix(".metadata.json").is_file():
                    stage = "validated_history_warm_start"
                else:
                    stage = "legacy_history_warm_start"
                if append_empty_core:
                    stage = "previous_core_count_warm_start"
                return plan, stage, None, str(path)
        return None

    def add_saved_candidate(
        source_ncores: int,
        append_empty_core: bool = False,
    ) -> None:
        saved = load_saved_candidate(
            problem,
            source_ncores,
            append_empty_core=append_empty_core,
        )
        if saved is not None:
            plan, stage, placement_diagnostics, source_plan = saved
            add_plan_factory(
                plan,
                stage,
                placement_diagnostics,
                source_plan,
            )

    if problem == 3:
        problem_2_counts = schedule_b.candidate_counts(analysis, ncores)
        validated_problem2 = load_saved_candidate(
            2, ncores, validated_only=True
        )
        if validated_problem2 is not None:
            plan, _, placement_diagnostics, source_plan = validated_problem2
            add_plan_factory(
                plan,
                "problem2_validated_warm_start",
                placement_diagnostics,
                source_plan,
            )
        else:
            if problem_2_counts:
                add_generated_factory(
                    f"problem2_warm_start_g{problem_2_counts[0]}",
                    generate_problem2_with_scoring,
                    problem_2_counts[0],
                )
    else:
        problem_2_counts = []
        add_saved_candidate(ncores)

    first_native_count = 0
    if group_counts:
        first_native_count = group_counts[0]
        add_generated_factory(
            f"current_{method}_g{first_native_count}",
            generate_with_scoring,
            first_native_count,
        )
        for topology_strategy in ("critical_path", "release_bytes"):
            add_generated_factory(
                f"topology_{topology_strategy}_g{first_native_count}",
                generate_with_scoring,
                first_native_count,
                topology_strategy,
            )

    if problem == 3:
        add_saved_candidate(ncores)
    if ncores > 1:
        add_saved_candidate(ncores - 1, append_empty_core=True)

    for group_count in group_counts:
        if group_count == first_native_count:
            continue
        add_generated_factory(
            f"current_{method}_g{group_count}",
            generate_with_scoring,
            group_count,
        )

    if problem == 3:
        for group_count in problem_2_counts[1:]:
            add_generated_factory(
                f"problem2_warm_start_g{group_count}",
                generate_problem2_with_scoring,
                group_count,
            )

    if len(analysis.eligible_ops) > 1:
        add_generated_factory(
            f"single_group_fallback_{method}",
            generate_with_scoring,
            1,
        )

    improvement_reserve = min(2, max_evals // 4) if improve and max_evals >= 6 else 0
    initial_eval_limit = max(1, max_evals - improvement_reserve)
    if candidate_selection == "ordered":
        for factory_stage, candidate_factory in candidate_factories:
            if attempted_evals >= initial_eval_limit or time.monotonic() >= deadline:
                break
            try:
                plan, stage, placement_diagnostics, source_plan = candidate_factory()
            except (TimeoutError, ValueError, RuntimeError) as error:
                errors.append(f"{factory_stage}: {error}")
                continue
            try:
                try_candidate(plan, stage, placement_diagnostics, source_plan)
            except (ValueError, RuntimeError) as error:
                errors.append(f"{stage}: {error}")
    else:
        proxy_entries: list[dict[str, Any]] = []
        for index, (factory_stage, candidate_factory) in enumerate(candidate_factories):
            if time.monotonic() >= deadline:
                break
            try:
                spec = candidate_factory()
            except (TimeoutError, ValueError, RuntimeError) as error:
                errors.append(f"{factory_stage}: {error}")
                continue
            plan, stage, placement_diagnostics, source_plan = spec
            metrics = None
            try:
                validate_plan(graph, plan, ncores)
                metrics = _plan_schedule_metrics(graph, plan, analysis)
            except (KeyError, ValueError, RuntimeError) as error:
                errors.append(f"{stage}: proxy metrics unavailable: {error}")
            proxy_entries.append({
                "index": index,
                "spec": spec,
                "metrics": metrics,
                "placement_diagnostics": placement_diagnostics,
            })

        selected_indices = _select_proxy_candidates(
            proxy_entries,
            initial_eval_limit,
            proxy_max_candidates,
            proxy_boundary_ratio_limit,
        )
        for entry in proxy_entries:
            plan, stage, placement_diagnostics, source_plan = entry["spec"]
            selection_metadata = entry["selection_metadata"]
            evaluate_candidate = entry["index"] in selected_indices
            if evaluate_candidate and attempted_evals >= initial_eval_limit:
                break
            try:
                try_candidate(
                    plan,
                    stage,
                    placement_diagnostics,
                    source_plan,
                    selection_metadata=selection_metadata,
                    evaluate_candidate=evaluate_candidate,
                )
            except (ValueError, RuntimeError) as error:
                errors.append(f"{stage}: {error}")

    if improve and best_result is not None and attempted_evals < max_evals:
        improved = True
        rounds = 0
        while improved and rounds < 2 and attempted_evals < max_evals:
            improved = False
            rounds += 1
            current = best_plan
            if current is None or time.monotonic() >= deadline:
                break
            parent_candidate_id = best_candidate_id
            for candidate, family in generate_neighbor_candidates(
                analysis, current, problem,
                limit=min(16, max_evals - attempted_evals),
                residence_ordering=residence_ordering == "neighbor",
                capacity=config.get("capacity", {}),
            ):
                if attempted_evals >= max_evals or time.monotonic() >= deadline:
                    break
                improved = try_candidate(
                    candidate,
                    f"local{rounds}_{family}",
                    parent_candidate=parent_candidate_id,
                    selection_metadata={
                        "proxy_selected": True,
                        "candidate_status": "local_improvement",
                        "proxy_selection_reasons": ["local_improvement"],
                    },
                ) or improved

    search_wall_sec = time.monotonic() - search_started
    solver_overhead_sec = max(0.0, search_wall_sec - multicore_eval_runtime)
    timing_fields = {
        "baseline_wall_sec": baseline_wall_sec,
        "search_wall_sec": search_wall_sec,
        "candidate_eval_wall_sec": multicore_eval_runtime,
        "solver_overhead_sec": solver_overhead_sec,
        "total_wall_sec": time.monotonic() - total_started,
    }
    evaluator.timeout_sec = per_candidate_timeout
    summary_selection_fields = {
        field: (
            best_selection_metadata.get(field)
            if best_selection_metadata is not None else None
        )
        for field in (
            "proxy_score", "proxy_rank", "pareto_kept", "proxy_selected",
            "candidate_status", "proxy_selection_reasons", "safety_status",
        )
    }

    if best_plan is None:
        if legal_draft is None:
            failure_status = (
                "not_run" if time.monotonic() >= deadline else "no_legal_candidate"
            )
        elif any("timeout" in error.lower() for error in errors):
            failure_status = "timeout"
        elif attempted_evals:
            failure_status = "evaluation_failed"
        else:
            failure_status = "not_run"
        row = {
            **run_fields,
            "case": graph.path.stem, "problem": problem, "ncores": ncores,
            "method": method, "makespan": None,
            "candidate_id": best_candidate_id,
            "best_candidate_source": best_candidate_source,
            "single_core_baseline": baseline_makespan, "speedup": None,
            "added_copy_bytes": None, "original_copy_bytes": None,
            "scheduled_copy_bytes": None, "cache_hit_rate": None,
            "subgraph_count": (
                len(set(legal_draft["node_to_subgraph"].values()))
                if legal_draft else 0
            ),
            "solver_runtime_sec": solver_overhead_sec,
            "evaluator_runtime_sec": baseline_runtime + multicore_eval_runtime,
            "baseline_cache_hit": baseline_cache_hit,
            "baseline_evaluator_runtime_sec": baseline_runtime,
            "evaluator_calls": evaluator.calls - calls_before, "seed": seed,
            "makespan_no_l2": None, "makespan_l2": None,
            "cache_speedup": None,
            "status": failure_status,
            "error": " | ".join(errors),
            **summary_selection_fields,
            **timing_fields,
        }
        _store_summary(results_dir, row)
        return row

    movement = best_result.get("data_movement_bytes", {}) if best_result else {}
    makespan = best_result.get("makespan") if best_result else None
    speedup = (
        baseline_makespan / makespan
        if baseline_makespan is not None and makespan not in (None, 0)
        else None
    )
    l2_speedup = None
    makespan_no_l2 = None
    makespan_l2 = makespan if problem == 3 else None
    if problem == 3:
        summary_path = results_dir / "summary.json"
        if summary_path.is_file():
            for item in json.loads(summary_path.read_text(encoding="utf-8")):
                if item.get("case") == graph.path.stem and item.get("problem") == 2 and item.get("ncores") == ncores:
                    if any(item.get(key) != value for key, value in {
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "graph_hash": fingerprints["graph_hash"],
                        "config_hash": fingerprints["config_hash"],
                        "official_code_hash": fingerprints["official_code_hash"],
                        "solver_code_hash": fingerprints["solver_code_hash"],
                        "seed": seed,
                        "time_limit_sec": time_limit,
                        "max_evals": max_evals,
                    }.items()):
                        continue
                    makespan_no_l2 = item.get("makespan")
                    break
        if makespan_no_l2 not in (None, 0) and makespan is not None:
            l2_speedup = makespan_no_l2 / makespan

    evaluator_runtime = baseline_runtime + multicore_eval_runtime
    row = {
        **run_fields,
        "case": graph.path.stem,
        "problem": problem,
        "ncores": ncores,
        "method": method,
        "candidate_id": best_candidate_id,
        "best_candidate_source": best_candidate_source,
        "makespan": makespan,
        "single_core_baseline": baseline_makespan,
        "speedup": speedup,
        "added_copy_bytes": movement.get("added_copy_bytes"),
        "original_copy_bytes": movement.get("original_graph_copy_bytes"),
        "scheduled_copy_bytes": movement.get("scheduled_copy_bytes"),
        "cache_hit_rate": (best_result.get("cache_stats", {}).get("hit_rate")
                            if best_result else None),
        "subgraph_count": len(set(best_plan["node_to_subgraph"].values())),
        "solver_runtime_sec": solver_overhead_sec,
        "evaluator_runtime_sec": evaluator_runtime,
        "baseline_cache_hit": baseline_cache_hit,
        "baseline_evaluator_runtime_sec": baseline_runtime,
        "evaluator_calls": evaluator.calls - calls_before,
        "seed": seed,
        "makespan_no_l2": makespan_no_l2,
        "makespan_l2": makespan_l2,
        "cache_speedup": l2_speedup,
        "status": "evaluated" if best_result else "legal_not_evaluated",
        "error": " | ".join(errors),
        "best_evaluation_tag": best_tag,
        **summary_selection_fields,
        **timing_fields,
    }
    _store_summary(results_dir, row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate, validate, and evaluate a multi-core schedule.")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--problem", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("-n", "--ncores", "--num-cores", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--baseline-timeout", type=float, default=300.0)
    parser.add_argument("--retry-baseline", action="store_true")
    parser.add_argument("--max-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-improve", action="store_true")
    parser.add_argument(
        "--placement-scoring", choices=("baseline", "soft"),
        default="baseline",
    )
    parser.add_argument(
        "--cache-ordering", choices=("fifo", "reuse_distance"),
        default="fifo",
    )
    parser.add_argument(
        "--candidate-selection", choices=("ordered", "proxy_pareto"),
        default="ordered",
    )
    parser.add_argument(
        "--residence-ordering", choices=("off", "neighbor"), default="off"
    )
    parser.add_argument("--proxy-max-candidates", type=int, default=4)
    parser.add_argument("--proxy-boundary-ratio-limit", type=float, default=2.0)
    parser.add_argument("--experiment-id", default="round2")
    parser.add_argument("--run-id")
    parser.add_argument("--official-root", type=Path, default=Path(__file__).resolve().parents[1] / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--history-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if (args.time_limit <= 0 or args.baseline_timeout <= 0
            or args.evaluator_timeout <= 0 or args.max_evals < 1
            or args.proxy_max_candidates < 1
            or args.proxy_boundary_ratio_limit <= 0):
        parser.error("time limits must be positive; max-evals must be at least 1")
    if not valid_run_label(args.experiment_id):
        parser.error("experiment-id must be a path-safe label")
    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")

    graph = load_graph(args.graph)
    config = load_config(args.official_root, args.config)
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else Path(__file__).resolve().parents[1]
        / "results_round2" / args.experiment_id / run_id
    )
    fingerprints = build_fingerprints(
        graph.path,
        config["path"],
        args.official_root.resolve(),
        Path(__file__).resolve().parent,
    )
    run_metadata = make_run_metadata(
        args.experiment_id,
        run_id,
        fingerprints,
        {
            "seed": args.seed,
            "time_limit_sec": args.time_limit,
            "baseline_timeout_sec": args.baseline_timeout,
            "evaluator_timeout_sec": args.evaluator_timeout,
            "max_evals": args.max_evals,
            "improve_enabled": not args.no_improve,
            "retry_baseline": args.retry_baseline,
            "placement_scoring": args.placement_scoring,
            "cache_ordering": args.cache_ordering,
            "candidate_selection": args.candidate_selection,
            "residence_ordering": args.residence_ordering,
            "proxy_max_candidates": args.proxy_max_candidates,
            "proxy_boundary_ratio_limit": args.proxy_boundary_ratio_limit,
            "history_dir": str(args.history_dir.resolve()),
        },
    )
    try:
        ensure_run_metadata(results_dir / "run_metadata.json", run_metadata)
    except ValueError as error:
        parser.error(str(error))
    manifest_path = results_dir / "batch_manifest.json"
    repository_root = Path(__file__).resolve().parents[1]
    data_dir_label = (
        graph.path.parent.relative_to(repository_root).as_posix()
        if graph.path.parent.is_relative_to(repository_root)
        else graph.path.parent.as_posix()
    )
    expected_manifest = {
        "cases": [graph.path.stem],
        "problems": [args.problem],
        "ncores": args.ncores,
        "data_dir": data_dir_label,
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid batch manifest: {error}")
        if (graph.path.stem not in manifest.get("cases", [])
                or args.problem not in manifest.get("problems", [])
                or any(ncore not in manifest.get("ncores", []) for ncore in args.ncores)):
            parser.error("requested combination is outside the run batch manifest")
    else:
        _save_plan(manifest_path, expected_manifest)
    evaluator = Evaluator(
        args.official_root,
        config["path"],
        results_dir,
        args.evaluator_timeout,
    )
    for ncores in args.ncores:
        if not 1 <= ncores <= 5:
            parser.error("ncores values must be between 1 and 5")
        row = solve_one(
            graph, args.problem, ncores, config, evaluator,
            results_dir, args.time_limit, args.max_evals,
            args.seed, not args.no_improve, args.baseline_timeout,
            args.retry_baseline, args.experiment_id, run_id,
            args.history_dir.resolve(),
            placement_scoring=args.placement_scoring,
            cache_ordering=args.cache_ordering,
            candidate_selection=args.candidate_selection,
            residence_ordering=args.residence_ordering,
            proxy_max_candidates=args.proxy_max_candidates,
            proxy_boundary_ratio_limit=args.proxy_boundary_ratio_limit,
        )
        print(
            f"{row['case']} p{row['problem']} n{row['ncores']}: "
            f"status={row['status']} makespan={row['makespan']} "
            f"added_copy_bytes={row['added_copy_bytes']} "
            f"evals={row['evaluator_calls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())