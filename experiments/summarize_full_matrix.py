from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.summarize_chain_matrix import (
    SUCCESS,
    _best,
    _candidate_statistics,
    _p3_best,
    _strict_pair_statistics,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(row.get("status") == status for row in rows)
        for status in sorted({row.get("status") for row in rows})
    }


def _candidate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "by_status": _status_counts(rows),
        "by_arm_status": {
            f"{row.get('arm')}|{row.get('status')}": sum(
                item.get("arm") == row.get("arm")
                and item.get("status") == row.get("status")
                for item in rows
            )
            for row in rows
        },
        "by_core_status": {
            f"{row.get('ncores')}|{row.get('status')}": sum(
                item.get("ncores") == row.get("ncores")
                and item.get("status") == row.get("status")
                for item in rows
            )
            for row in rows
        },
    }


def _singlecore_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row.get("makespan") for row in rows if row.get("makespan") is not None]
    return {
        "total": len(rows),
        "by_status": _status_counts(rows),
        "successful_cases": len(values),
        "missing_cases": len(rows) - len(values),
        "makespan_min": min(values) if values else None,
        "makespan_max": max(values) if values else None,
    }


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _arm_comparison(
    chain: dict[str, Any] | None,
    contiguous: dict[str, Any] | None,
    prefix: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{prefix}_chain_makespan": chain.get("makespan") if chain else None,
        f"{prefix}_contiguous_makespan": (
            contiguous.get("makespan") if contiguous else None
        ),
        f"{prefix}_chain_group_count": chain.get("group_count") if chain else None,
        f"{prefix}_contiguous_group_count": (
            contiguous.get("group_count") if contiguous else None
        ),
        f"{prefix}_chain_plan_hash": chain.get("plan_hash") if chain else None,
        f"{prefix}_contiguous_plan_hash": (
            contiguous.get("plan_hash") if contiguous else None
        ),
    }
    if chain is None or contiguous is None:
        row.update({
            f"{prefix}_status": "missing",
            f"{prefix}_outcome": "not_evaluated",
            f"{prefix}_delta": None,
            f"{prefix}_chain_over_contiguous_ratio": None,
            f"{prefix}_speedup_vs_contiguous": None,
        })
        return row
    chain_value = float(chain["makespan"])
    contiguous_value = float(contiguous["makespan"])
    row.update({
        f"{prefix}_status": "complete",
        f"{prefix}_outcome": (
            "win" if chain_value < contiguous_value
            else "tie" if chain_value == contiguous_value
            else "loss"
        ),
        f"{prefix}_delta": chain_value - contiguous_value,
        f"{prefix}_chain_over_contiguous_ratio": (
            chain_value / contiguous_value if contiguous_value else None
        ),
        f"{prefix}_speedup_vs_contiguous": (
            contiguous_value / chain_value if chain_value else None
        ),
    })
    return row


def _problem_rows(
    p1_rows: list[dict[str, Any]],
    p2_rows: list[dict[str, Any]],
    p3_rows: list[dict[str, Any]],
    singlecore_rows: list[dict[str, Any]],
    cases: list[str],
    ncores: list[int],
) -> list[dict[str, Any]]:
    singlecore = {
        row["case"]: row
        for row in singlecore_rows
        if row.get("status") in SUCCESS and row.get("makespan") is not None
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        for core in ncores:
            p1_chain = _best(p1_rows, "chain_contiguous_p1", case, core)
            p1_contiguous = _best(p1_rows, "contiguous_p1", case, core)
            p2_chain = _best(p2_rows, "chain_contiguous", case, core)
            p2_contiguous = _best(p2_rows, "contiguous_p2", case, core)
            p3_chain = _p3_best(p3_rows, "chain_contiguous", case, core)
            p3_contiguous = _p3_best(p3_rows, "contiguous_p2", case, core)
            baseline = singlecore.get(case)
            row: dict[str, Any] = {"case": case, "ncores": core}
            row.update(_arm_comparison(p1_chain, p1_contiguous, "p1"))
            row.update(_arm_comparison(p2_chain, p2_contiguous, "p2"))
            row.update({
                "singlecore_status": baseline.get("status") if baseline else "missing",
                "singlecore_makespan": baseline.get("makespan") if baseline else None,
                "p1_chain_speedup_vs_singlecore": _ratio(
                    baseline.get("makespan") if baseline else None,
                    p1_chain.get("makespan") if p1_chain else None,
                ),
                "p1_contiguous_speedup_vs_singlecore": _ratio(
                    baseline.get("makespan") if baseline else None,
                    p1_contiguous.get("makespan") if p1_contiguous else None,
                ),
                "p2_chain_speedup_vs_singlecore": _ratio(
                    baseline.get("makespan") if baseline else None,
                    p2_chain.get("makespan") if p2_chain else None,
                ),
                "p2_contiguous_speedup_vs_singlecore": _ratio(
                    baseline.get("makespan") if baseline else None,
                    p2_contiguous.get("makespan") if p2_contiguous else None,
                ),
                "p3_chain_makespan": p3_chain.get("makespan_p3") if p3_chain else None,
                "p3_contiguous_makespan": (
                    p3_contiguous.get("makespan_p3") if p3_contiguous else None
                ),
                "p3_chain_plan_hash": p3_chain.get("plan_hash") if p3_chain else None,
                "p3_contiguous_plan_hash": (
                    p3_contiguous.get("plan_hash") if p3_contiguous else None
                ),
                "p3_chain_cache_hit_rate": (
                    p3_chain.get("cache_hit_rate") if p3_chain else None
                ),
                "p3_contiguous_cache_hit_rate": (
                    p3_contiguous.get("cache_hit_rate") if p3_contiguous else None
                ),
                "p3_case_core_pair_complete": (
                    p3_chain is not None and p3_contiguous is not None
                ),
                "p3_same_plan_hash_pair_complete": bool(
                    p3_chain is not None
                    and p3_contiguous is not None
                    and p3_chain.get("plan_hash") == p3_contiguous.get("plan_hash")
                ),
            })
            if p3_chain is None or p3_contiguous is None:
                row["p3_case_core_outcome"] = "not_evaluated"
            else:
                chain_value = float(p3_chain["makespan_p3"])
                contiguous_value = float(p3_contiguous["makespan_p3"])
                row["p3_case_core_outcome"] = (
                    "win" if chain_value < contiguous_value
                    else "tie" if chain_value == contiguous_value
                    else "loss"
                )
            rows.append(row)
    return rows


def _source_info(
    label: str,
    directory: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "directory": str(directory),
        "run_id": summary.get("run_id"),
        "matrix_spec_hash": summary.get("matrix_spec_hash"),
        "row_count": len(summary.get("comparisons", [])),
        "official_evaluation_calls": summary.get(
            "official_evaluation_calls",
            (manifest or {}).get("official_evaluation_calls", 0),
        ),
        "status_counts": _status_counts(summary.get("comparisons", [])),
    }


def summarize(
    p1: dict[str, Any],
    p2: dict[str, Any],
    p3: dict[str, Any],
    singlecore: dict[str, Any],
    source_info: list[dict[str, Any]],
) -> dict[str, Any]:
    p1_rows = p1.get("comparisons", [])
    p2_rows = p2.get("comparisons", [])
    p3_rows = p3.get("comparisons", [])
    singlecore_rows = singlecore.get("comparisons", [])
    specs = [p1.get("matrix_spec", {}), p2.get("matrix_spec", {})]
    cases = sorted({case for spec in specs for case in spec.get("cases", [])})
    cores = sorted({core for spec in specs for core in spec.get("ncores", [])})
    rows = _problem_rows(
        p1_rows, p2_rows, p3_rows, singlecore_rows, cases, cores
    )
    p1_stats = _candidate_stats(p1_rows)
    p2_stats = _candidate_stats(p2_rows)
    p3_stats = _candidate_stats(p3_rows)
    p3_strict = _strict_pair_statistics(p2_rows, p3_rows)
    p1_outcomes = {
        outcome: sum(row.get("p1_outcome") == outcome for row in rows)
        for outcome in ("win", "tie", "loss", "not_evaluated")
    }
    p2_outcomes = {
        outcome: sum(row.get("p2_outcome") == outcome for row in rows)
        for outcome in ("win", "tie", "loss", "not_evaluated")
    }
    p3_outcomes = {
        outcome: sum(row.get("p3_case_core_outcome") == outcome for row in rows)
        for outcome in ("win", "tie", "loss", "not_evaluated")
    }
    return {
        "coverage": {
            "case_count": len(cases),
            "cases": cases,
            "ncores": cores,
            "case_core_count": len(rows),
            "problem_1_candidate_count": len(p1_rows),
            "problem_2_candidate_count": len(p2_rows),
            "problem_3_candidate_count": len(p3_rows),
            "singlecore_case_count": len(singlecore_rows),
        },
        "sources": source_info,
        "candidate_statistics": {
            "problem_1": p1_stats,
            "problem_2": p2_stats,
            "problem_3": p3_stats,
            "singlecore": _singlecore_stats(singlecore_rows),
        },
        "case_core_outcomes": {
            "problem_1_chain_vs_contiguous": p1_outcomes,
            "problem_2_chain_vs_contiguous": p2_outcomes,
            "problem_3_chain_vs_contiguous_best": p3_outcomes,
        },
        "problem_3_strict_plan_pairing": p3_strict,
        "reserve_statistics": p2.get("reserve_statistics", {}),
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the complete P1/P2/P3 matrix.")
    parser.add_argument("--p1-results-dir", type=Path, required=True)
    parser.add_argument("--p2-results-dir", type=Path, required=True)
    parser.add_argument("--p3-results-dir", type=Path, required=True)
    parser.add_argument("--singlecore-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    p1_dir = args.p1_results_dir.resolve()
    p2_dir = args.p2_results_dir.resolve()
    p3_dir = args.p3_results_dir.resolve()
    singlecore_dir = args.singlecore_results_dir.resolve()
    output_dir = args.output_dir.resolve()
    p1 = _load(p1_dir / "phase4_matrix.json")
    p2 = _load(p2_dir / "phase4_matrix.json")
    p3 = _load(p3_dir / "selected_problem3.json")
    singlecore = _load(singlecore_dir / "singlecore_matrix.json")
    p1_manifest = (
        _load(p1_dir / "batch_manifest.json")
        if (p1_dir / "batch_manifest.json").is_file() else None
    )
    source_info = [
        _source_info("problem_1", p1_dir, p1, p1_manifest),
        _source_info("problem_2", p2_dir, p2),
        _source_info("problem_3", p3_dir, p3),
        _source_info("singlecore", singlecore_dir, singlecore),
    ]
    result = summarize(p1, p2, p3, singlecore, source_info)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "full_matrix_summary.json", result)
    _write_csv(output_dir / "full_case_core_summary.csv", result["rows"])
    _write_csv(
        output_dir / "full_strict_plan_pair_summary.csv",
        result["problem_3_strict_plan_pairing"]["p3_pair_rows"],
    )
    candidate_rows = []
    for problem, stats in result["candidate_statistics"].items():
        for status, count in stats["by_status"].items():
            candidate_rows.append({
                "problem": problem,
                "arm": None,
                "ncores": None,
                "status": status,
                "count": count,
            })
    _write_csv(output_dir / "full_candidate_status_summary.csv", candidate_rows)
    print(
        f"case_core_rows={len(result['rows'])} "
        f"p1={result['case_core_outcomes']['problem_1_chain_vs_contiguous']} "
        f"p2={result['case_core_outcomes']['problem_2_chain_vs_contiguous']} "
        f"p3={result['case_core_outcomes']['problem_3_chain_vs_contiguous_best']} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())