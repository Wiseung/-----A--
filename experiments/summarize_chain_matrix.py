from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUCCESS = {"success", "equivalent_plan"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _objective(row: dict[str, Any]) -> tuple[float, float, float]:
    return tuple(
        float(row.get(field)) if row.get(field) is not None else float("inf")
        for field in ("makespan", "added_copy_bytes", "spill_added_copy_bytes")
    )


def _best(
    rows: list[dict[str, Any]],
    arm: str,
    case: str,
    ncores: int,
    makespan_field: str = "makespan",
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("arm") == arm
        and row.get("case") == case
        and row.get("ncores") == ncores
        and row.get("status") in SUCCESS
        and row.get(makespan_field) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=_objective)


def _p3_best(
    rows: list[dict[str, Any]],
    arm: str,
    case: str,
    ncores: int,
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("arm") == arm
        and row.get("case") == case
        and row.get("ncores") == ncores
        and row.get("status") in SUCCESS
        and row.get("makespan_p3") is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row.get("makespan_p3")),
            float(row.get("added_copy_bytes_p3") or float("inf")),
            float(row.get("spill_added_copy_bytes_p3") or float("inf")),
        ),
    )


def _counts_by(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = "|".join(str(row.get(field)) for field in fields)
        result[key] = result.get(key, 0) + 1
    return result


def _candidate_statistics(
    p2_rows: list[dict[str, Any]],
    p3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "problem_2": {
            "total": len(p2_rows),
            "by_status": _counts_by(p2_rows, ("status",)),
            "by_arm_status": _counts_by(p2_rows, ("arm", "status")),
            "by_core_status": _counts_by(p2_rows, ("ncores", "status")),
            "by_source_status": _counts_by(
                p2_rows, ("source_results_dir", "status")
            ),
        },
        "problem_3": {
            "total": len(p3_rows),
            "by_status": _counts_by(p3_rows, ("status",)),
            "by_arm_status": _counts_by(p3_rows, ("arm", "status")),
            "by_core_status": _counts_by(p3_rows, ("ncores", "status")),
            "by_source_status": _counts_by(
                p3_rows, ("source_results_dir", "status")
            ),
        },
    }


def _strict_plan_pairs(
    p2_rows: list[dict[str, Any]],
    p3_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    p2_by_hash: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in p2_rows:
        if row.get("status") in SUCCESS and row.get("plan_hash"):
            p2_by_hash.setdefault(
                (row["case"], row["ncores"], row["plan_hash"]), []
            ).append(row)
    p3_by_hash: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]] = {}
    for row in p3_rows:
        if row.get("status") in SUCCESS and row.get("plan_hash"):
            p3_by_hash.setdefault(
                (row["case"], row["ncores"], row["plan_hash"]), {}
            ).setdefault(row.get("arm"), []).append(row)

    pairs: list[dict[str, Any]] = []
    for key, arm_rows in sorted(p3_by_hash.items()):
        p2_candidates = p2_by_hash.get(key, [])
        for arm, rows in sorted(arm_rows.items()):
            p3 = min(rows, key=lambda row: (
                float(row.get("makespan_p3"))
                if row.get("makespan_p3") is not None else float("inf"),
                float(row.get("added_copy_bytes_p3"))
                if row.get("added_copy_bytes_p3") is not None else float("inf"),
            ))
            p2_same_arm = [row for row in p2_candidates if row.get("arm") == arm]
            p2 = min(p2_same_arm or p2_candidates, key=_objective, default=None)
            p2_value = p2.get("makespan") if p2 else None
            p3_value = p3.get("makespan_p3")
            row = {
                "case": key[0],
                "ncores": key[1],
                "plan_hash": key[2],
                "arm": arm,
                "group_count": p3.get("group_count"),
                "p2_arm": p2.get("arm") if p2 else None,
                "p2_candidate_id": p2.get("candidate_id") if p2 else None,
                "p3_candidate_id": p3.get("candidate_id"),
                "p3_evaluation_source_candidate_id": p3.get(
                    "evaluation_source_candidate_id"
                ),
                "p3_same_hash_contiguous_available": (
                    "contiguous_p2" in arm_rows
                ),
                "p3_same_hash_chain_available": (
                    "chain_contiguous" in arm_rows
                ),
                "p3_same_hash_missing_arm": (
                    "contiguous_p2"
                    if arm == "chain_contiguous" and "contiguous_p2" not in arm_rows
                    else "chain_contiguous"
                    if arm == "contiguous_p2" and "chain_contiguous" not in arm_rows
                    else None
                ),
                "p2_status": p2.get("status") if p2 else "missing",
                "p3_status": p3.get("status"),
                "p2_makespan": p2_value,
                "p3_makespan": p3_value,
                "p2_to_p3_delta": (
                    p3_value - p2_value
                    if p2_value is not None and p3_value is not None else None
                ),
                "p3_over_p2_ratio": (
                    p3_value / p2_value
                    if p2_value not in (None, 0) and p3_value is not None else None
                ),
                "cache_hit_bytes": p3.get("cache_hit_bytes"),
                "cache_miss_bytes": p3.get("cache_miss_bytes"),
                "cache_hit_rate": p3.get("cache_hit_rate"),
                "partition_added_copy_bytes": p3.get(
                    "partition_added_copy_bytes_p3"
                ),
                "spill_added_copy_bytes": p3.get("spill_added_copy_bytes_p3"),
                "p2_source_results_dir": p2.get("source_results_dir") if p2 else None,
                "p3_source_results_dir": p3.get("source_results_dir"),
            }
            row["p2_p3_pair_complete"] = p2 is not None
            row["p2_p3_candidate_source_differs"] = bool(
                p2 and p3.get("candidate_id")
                and p2.get("candidate_id") != p3.get("candidate_id")
            )
            row["source_mismatch"] = bool(
                row["p2_source_results_dir"]
                and row["p3_source_results_dir"]
                and row["p2_source_results_dir"] != row["p3_source_results_dir"]
            )
            pairs.append(row)
    return pairs


def _strict_pair_statistics(
    p2_rows: list[dict[str, Any]],
    p3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = _strict_plan_pairs(p2_rows, p3_rows)
    groups: dict[tuple[str, int, str], set[str | None]] = {}
    for row in p3_rows:
        if row.get("status") in SUCCESS and row.get("plan_hash"):
            groups.setdefault(
                (row["case"], row["ncores"], row["plan_hash"]),
                set(),
            ).add(row.get("arm"))
    complete = [row for row in pairs if row["p2_p3_pair_complete"]]
    unique_p3: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in p3_rows:
        if row.get("status") in SUCCESS and row.get("plan_hash"):
            unique_p3.setdefault(
                (row["case"], row["ncores"], row["plan_hash"]), row
            )
    total_hit = sum(int(row.get("cache_hit_bytes") or 0) for row in unique_p3.values())
    total_miss = sum(int(row.get("cache_miss_bytes") or 0) for row in unique_p3.values())
    source_candidate_diffs = sum(
        row["p2_p3_candidate_source_differs"] for row in complete
    )
    candidate_ids_by_key: dict[
        tuple[str, int, str], dict[str, set[str]]
    ] = {}
    for row in p3_rows:
        if row.get("status") not in SUCCESS or not row.get("plan_hash"):
            continue
        key = (row["case"], row["ncores"], row["plan_hash"])
        candidate_id = row.get("candidate_id")
        if candidate_id:
            candidate_ids_by_key.setdefault(key, {}).setdefault(
                row.get("arm"), set()
            ).add(candidate_id)
    hash_group_rows = []
    for key, arm_rows in sorted(groups.items()):
        candidate_ids = {
            arm: sorted(ids)
            for arm, ids in sorted(candidate_ids_by_key.get(key, {}).items())
        }
        present = sorted(arm_rows)
        missing = sorted({"contiguous_p2", "chain_contiguous"} - set(present))
        hash_group_rows.append({
            "case": key[0],
            "ncores": key[1],
            "plan_hash": key[2],
            "arms": present,
            "missing_arm": missing[0] if len(missing) == 1 else None,
            "candidate_ids_by_arm": candidate_ids,
            "candidate_source_differs": len({
                candidate_id
                for ids in candidate_ids.values()
                for candidate_id in ids
            }) > 1,
        })
    return {
        "pair_count": len(pairs),
        "complete_p2_p3_pair_count": len(complete),
        "missing_p2_pair_count": len(pairs) - len(complete),
        "source_mismatch_count": sum(row["source_mismatch"] for row in pairs),
        "p2_p3_candidate_source_differs_count": source_candidate_diffs,
        "cache_hit_bytes_total": total_hit,
        "cache_miss_bytes_total": total_miss,
        "cache_hit_rate_weighted": (
            total_hit / (total_hit + total_miss)
            if total_hit + total_miss else None
        ),
        "p3_plan_hash_group_count": len(groups),
        "p3_only_chain_plan_count": sum(
            arms == {"chain_contiguous"} for arms in groups.values()
        ),
        "p3_only_contiguous_plan_count": sum(
            arms == {"contiguous_p2"} for arms in groups.values()
        ),
        "p3_complete_arm_pair_count": sum(
            arms == {"contiguous_p2", "chain_contiguous"}
            for arms in groups.values()
        ),
        "p3_hash_groups_with_multiple_candidate_ids": sum(
            item["candidate_source_differs"] for item in hash_group_rows
        ),
        "p3_hash_groups": hash_group_rows,
        "p3_pair_rows": pairs,
    }


def _comparison(chain: dict[str, Any] | None, baseline: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if chain is None or baseline is None:
        return {
            f"{prefix}_status": "missing",
            f"{prefix}_delta": None,
            f"{prefix}_outcome": "not_evaluated",
        }
    chain_value = float(chain["makespan_p3"] if prefix == "p3" else chain["makespan"])
    baseline_value = float(baseline["makespan_p3"] if prefix == "p3" else baseline["makespan"])
    return {
        f"{prefix}_status": "complete",
        f"{prefix}_delta": chain_value - baseline_value,
        f"{prefix}_delta_ratio": chain_value / baseline_value - 1.0 if baseline_value else None,
        f"{prefix}_outcome": (
            "win" if chain_value < baseline_value
            else "tie" if chain_value == baseline_value
            else "loss"
        ),
        f"{prefix}_chain_group_count": chain.get("group_count"),
        f"{prefix}_baseline_group_count": baseline.get("group_count"),
        f"{prefix}_chain_plan_hash": chain.get("plan_hash"),
        f"{prefix}_baseline_plan_hash": baseline.get("plan_hash"),
        f"{prefix}_chain_partition_copy_bytes": chain.get(
            "partition_added_copy_bytes_p3" if prefix == "p3" else "partition_added_copy_bytes"
        ),
        f"{prefix}_baseline_partition_copy_bytes": baseline.get(
            "partition_added_copy_bytes_p3" if prefix == "p3" else "partition_added_copy_bytes"
        ),
        f"{prefix}_chain_spill_bytes": chain.get(
            "spill_added_copy_bytes_p3" if prefix == "p3" else "spill_added_copy_bytes"
        ),
        f"{prefix}_baseline_spill_bytes": baseline.get(
            "spill_added_copy_bytes_p3" if prefix == "p3" else "spill_added_copy_bytes"
        ),
    }


def summarize(
    p2_summary: dict[str, Any],
    p3_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    p2_rows = p2_summary.get("comparisons", [])
    p3_rows = p3_summary.get("comparisons", []) if p3_summary else []
    spec = p2_summary.get("matrix_spec", {})
    cases = spec.get("cases", sorted({row.get("case") for row in p2_rows}))
    cores = spec.get("ncores", sorted({row.get("ncores") for row in p2_rows}))
    rows = []
    for case in cases:
        for ncores in cores:
            chain_p2 = _best(p2_rows, "chain_contiguous", case, ncores)
            base_p2 = _best(p2_rows, "contiguous_p2", case, ncores)
            chain_p3 = _p3_best(p3_rows, "chain_contiguous", case, ncores)
            base_p3 = _p3_best(p3_rows, "contiguous_p2", case, ncores)
            row = {"case": case, "ncores": ncores}
            row.update(_comparison(chain_p2, base_p2, "p2"))
            row.update(_comparison(chain_p3, base_p3, "p3"))
            row["p3_case_core_pair_complete"] = (
                chain_p3 is not None and base_p3 is not None
            )
            row["p3_same_plan_hash_pair_complete"] = bool(
                chain_p3 is not None
                and base_p3 is not None
                and chain_p3.get("plan_hash") == base_p3.get("plan_hash")
            )
            row["p3_pair_complete"] = row["p3_case_core_pair_complete"]
            row["p3_missing_chain"] = chain_p3 is None
            row["p3_missing_contiguous"] = base_p3 is None
            if chain_p2 is not None:
                row.update({
                    "p2_chain_active_core_count": chain_p2.get("active_core_count"),
                    "p2_chain_cache_hit_rate": chain_p2.get("cache_hit_rate"),
                    "p2_chain_safety_status": chain_p2.get("safety_status"),
                    "p2_chain_candidate_status": chain_p2.get("candidate_status"),
                })
            if chain_p3 is not None:
                row.update({
                    "p3_chain_cache_hit_rate": chain_p3.get("cache_hit_rate"),
                })
            rows.append(row)
    case_core_p3_rows = [
        row for row in rows if row.get("p3_case_core_pair_complete")
    ]
    strict_plan_pairing = _strict_pair_statistics(p2_rows, p3_rows)
    case_core_pairing = {
        "complete_pair_count": len(case_core_p3_rows),
        "missing_chain_count": sum(row["p3_missing_chain"] for row in rows),
        "missing_contiguous_count": sum(
            row["p3_missing_contiguous"] for row in rows
        ),
        "same_plan_hash_pair_count": sum(
            row["p3_same_plan_hash_pair_complete"] for row in rows
        ),
        "outcomes": {
            outcome: sum(row.get("p3_outcome") == outcome for row in case_core_p3_rows)
            for outcome in ("win", "tie", "loss")
        },
        "case_core_set": [
            {"case": row["case"], "ncores": row["ncores"]}
            for row in case_core_p3_rows
        ],
    }
    return {
        "p2_matrix_spec": spec,
        "p3_available": bool(p3_rows),
        "row_count": len(rows),
        "p2_outcomes": {
            outcome: sum(row.get("p2_outcome") == outcome for row in rows)
            for outcome in ("win", "tie", "loss", "not_evaluated")
        },
        "p3_outcomes": {
            outcome: sum(row.get("p3_outcome") == outcome for row in rows)
            for outcome in ("win", "tie", "loss", "not_evaluated")
        },
        "p3_case_core_pairing": case_core_pairing,
        "reserve_statistics": p2_summary.get("reserve_statistics", {}),
        "candidate_statistics": _candidate_statistics(p2_rows, p3_rows),
        "strict_plan_pairing": strict_plan_pairing,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize chain risk-retention P2/P3 results.")
    parser.add_argument("--p2-results-dir", type=Path, required=True)
    parser.add_argument("--p3-results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    p2_dir = args.p2_results_dir.resolve()
    p3_dir = args.p3_results_dir.resolve() if args.p3_results_dir else None
    output_dir = (args.output_dir or p2_dir).resolve()
    p2_path = p2_dir / "phase4_matrix.json"
    if not p2_path.is_file():
        parser.error(f"missing P2 result: {p2_path}")
    p3_path = p3_dir / "selected_problem3.json" if p3_dir else None
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    p3 = json.loads(p3_path.read_text(encoding="utf-8")) if p3_path and p3_path.is_file() else None
    result = summarize(p2, p3)
    _write_json(output_dir / "chain_risk_retention_summary.json", result)
    if result["rows"]:
        fields = []
        for row in result["rows"]:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with (output_dir / "chain_risk_retention_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["rows"])
    strict_rows = result["strict_plan_pairing"]["p3_pair_rows"]
    if strict_rows:
        fields = []
        for row in strict_rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with (output_dir / "strict_plan_pair_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(strict_rows)
    print(
        f"rows={result['row_count']} p2={result['p2_outcomes']} "
        f"p3={result['p3_outcomes']} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())