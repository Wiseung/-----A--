from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUCCESS = {"success", "equivalent_plan"}


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


def _matrix_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ordered_union(values: list[list[Any]]) -> list[Any]:
    result: list[Any] = []
    for group in values:
        for value in group:
            if value not in result:
                result.append(value)
    return result


def _merged_spec(specs: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "cases": sorted({case for spec in specs for case in spec.get("cases", [])}),
        "ncores": sorted({core for spec in specs for core in spec.get("ncores", [])}),
        "group_counts": sorted({
            count for spec in specs for count in spec.get("group_counts", [])
        }),
        "arms": _ordered_union([spec.get("arms", []) for spec in specs]),
        "evaluation_problems": sorted({
            problem
            for spec in specs
            for problem in spec.get("evaluation_problems", [])
        }),
    }
    scalar_fields = (
        "schedule_problem",
        "cache_ordering",
        "placement_scoring_override",
        "max_proxy_candidates",
        "boundary_ratio_limit",
        "proxy_screen",
    )
    for field in scalar_fields:
        values = [spec[field] for spec in specs if field in spec]
        if values and all(value == values[0] for value in values):
            merged[field] = values[0]
        elif values:
            merged[field] = None
    return merged


def _aggregate_reserve(stats: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "high_risk_reserved_count": 0,
        "high_risk_evaluated_count": 0,
        "high_risk_not_run_count": 0,
        "high_risk_wins": 0,
        "high_risk_ties": 0,
        "high_risk_losses": 0,
        "high_risk_outcomes": [],
    }
    for item in stats:
        for field in (
            "high_risk_reserved_count",
            "high_risk_evaluated_count",
            "high_risk_not_run_count",
            "high_risk_wins",
            "high_risk_ties",
            "high_risk_losses",
        ):
            result[field] += int(item.get(field, 0) or 0)
        result["high_risk_outcomes"].extend(item.get("high_risk_outcomes", []))
    evaluated = result["high_risk_evaluated_count"]
    result["high_risk_recovery_rate"] = (
        result["high_risk_wins"] / evaluated if evaluated else None
    )
    return result


def merge(source_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    rows: list[dict[str, Any]] = []
    source_specs: list[dict[str, Any]] = []
    source_info: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    reserve_stats: list[dict[str, Any]] = []
    official_calls = 0

    for source_dir in source_dirs:
        source_dir = source_dir.resolve()
        source_path = source_dir / "phase4_matrix.json"
        if not source_path.is_file():
            raise ValueError(f"missing phase4_matrix.json: {source_path}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        manifest_path = source_dir / "batch_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        spec = source.get("matrix_spec", {})
        source_specs.append(spec)
        reserve_stats.append(source.get("reserve_statistics", {}))
        source_calls = int(
            manifest.get("official_evaluation_calls", source.get("official_evaluation_calls", 0))
            or 0
        )
        official_calls += source_calls
        source_info.append({
            "results_dir": _portable_path(source_dir),
            "run_id": source.get("run_id"),
            "experiment_id": source.get("experiment_id"),
            "matrix_spec_hash": source.get("matrix_spec_hash"),
            "row_count": len(source.get("comparisons", [])),
            "official_evaluation_calls": source_calls,
        })
        for source_row in source.get("comparisons", []):
            key = (
                source_row.get("case"),
                source_row.get("ncores"),
                source_row.get("arm"),
                source_row.get("group_count"),
                source_row.get("evaluation_problem"),
            )
            if key in seen_keys:
                raise ValueError(f"duplicate matrix row key: {key}")
            seen_keys.add(key)
            row = dict(source_row)
            row.update({
                "source_results_dir": _portable_path(source_dir),
                "source_run_id": source.get("run_id"),
                "source_matrix_spec_hash": source.get("matrix_spec_hash"),
            })
            if row.get("status") in SUCCESS:
                plan_path = Path(row.get("plan_path", ""))
                if not plan_path.is_absolute():
                    plan_path = ROOT / plan_path
                if not plan_path.is_file():
                    raise ValueError(f"successful row has missing plan: {plan_path}")
                if not row.get("plan_hash"):
                    raise ValueError(f"successful row has missing plan_hash: {key}")
            rows.append(row)

    merged_spec = _merged_spec(source_specs)
    rows.sort(key=lambda row: (
        row.get("case", ""),
        int(row.get("ncores", 0)),
        str(row.get("arm", "")),
        int(row.get("group_count", 0)),
        int(row.get("evaluation_problem", 0)),
    ))
    matrix_hash = _matrix_hash({
        "matrix_spec": merged_spec,
        "sources": source_info,
        "row_keys": [
            [
                row.get("case"), row.get("ncores"), row.get("arm"),
                row.get("group_count"), row.get("evaluation_problem"),
                row.get("plan_hash"),
            ]
            for row in rows
        ],
    })
    result = {
        "experiment_id": "r05_merged_phase4_matrix",
        "run_id": output_dir.name,
        "matrix_spec": merged_spec,
        "matrix_spec_hash": matrix_hash,
        "sources": source_info,
        "source_matrix_specs": source_specs,
        "comparisons": rows,
        "completed_evaluation_count": len(rows),
        "official_evaluation_calls": official_calls,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status") for row in rows})
        },
        "reserve_statistics": _aggregate_reserve(reserve_stats),
    }
    _write_json(output_dir / "phase4_matrix.json", result)
    _write_json(output_dir / "merge_manifest.json", {
        "status": "completed",
        "output_dir": _portable_path(output_dir),
        "source_count": len(source_dirs),
        "sources": source_info,
        "row_count": len(rows),
        "official_evaluation_calls": official_calls,
        "matrix_spec_hash": matrix_hash,
        "status_counts": result["status_counts"],
        "reserve_statistics": result["reserve_statistics"],
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge completed Phase 4 matrix results.")
    parser.add_argument("--source-results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = merge(
            [path.resolve() for path in args.source_results_dir],
            args.output_dir.resolve(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        f"rows={result['completed_evaluation_count']} "
        f"official_calls={result['official_evaluation_calls']} "
        f"status_counts={result['status_counts']} output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())