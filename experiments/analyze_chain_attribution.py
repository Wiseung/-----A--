from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    return float(value) if value is not None else None


def _sort_value(row: dict[str, Any], field: str) -> float:
    value = _number(row, field)
    return value if value is not None else float("inf")


def _max_core_load(row: dict[str, Any], field: str) -> int:
    values = row.get(field) or {}
    return max((int(value) for value in values.values()), default=0)


def _select_rows(
    rows: list[dict[str, Any]],
    arm: str,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    selected: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("arm") != arm or row.get("evaluation_problem") != 2:
            continue
        if row.get("status") not in {"success", "equivalent_plan"}:
            continue
        key = (row["case"], row["ncores"], row["group_count"])
        previous = selected.get(key)
        if previous is None or (
            _sort_value(row, "makespan"),
            _sort_value(row, "added_copy_bytes"),
        ) < (
            _sort_value(previous, "makespan"),
            _sort_value(previous, "added_copy_bytes"),
        ):
            selected[key] = row
    return selected


def _comparison(
    baseline: dict[str, Any],
    chain: dict[str, Any],
) -> dict[str, Any]:
    makespan_delta = float(chain["makespan"]) - float(baseline["makespan"])
    boundary_delta = (
        float(chain["partition_added_copy_bytes"])
        - float(baseline["partition_added_copy_bytes"])
    )
    spill_delta = (
        float(chain.get("spill_added_copy_bytes") or 0)
        - float(baseline.get("spill_added_copy_bytes") or 0)
    )
    if boundary_delta < 0 and makespan_delta < 0:
        classification = "A_boundary_down_makespan_down"
    elif boundary_delta < 0 and makespan_delta >= 0:
        classification = "B_boundary_down_makespan_not_down"
    elif boundary_delta > 0 and makespan_delta < 0:
        classification = "C_boundary_up_makespan_down"
    elif boundary_delta > 0 and makespan_delta >= 0:
        classification = "D_boundary_up_makespan_up"
    elif makespan_delta < 0:
        classification = "E_boundary_same_makespan_down"
    elif makespan_delta > 0:
        classification = "F_boundary_same_makespan_up"
    else:
        classification = "G_both_same"
    baseline_m = _max_core_load(baseline, "m_cycles_by_core")
    chain_m = _max_core_load(chain, "m_cycles_by_core")
    baseline_v = _max_core_load(baseline, "v_cycles_by_core")
    chain_v = _max_core_load(chain, "v_cycles_by_core")
    return {
        "case": baseline["case"],
        "ncores": baseline["ncores"],
        "group_count": baseline["group_count"],
        "baseline_makespan": baseline["makespan"],
        "chain_makespan": chain["makespan"],
        "makespan_delta": makespan_delta,
        "makespan_delta_ratio": makespan_delta / float(baseline["makespan"]),
        "baseline_partition_copy_bytes": baseline["partition_added_copy_bytes"],
        "chain_partition_copy_bytes": chain["partition_added_copy_bytes"],
        "partition_copy_delta": boundary_delta,
        "partition_copy_delta_ratio": boundary_delta / max(
            float(baseline["partition_added_copy_bytes"]), 1.0
        ),
        "baseline_spill_added_copy_bytes": baseline.get("spill_added_copy_bytes"),
        "chain_spill_added_copy_bytes": chain.get("spill_added_copy_bytes"),
        "spill_delta": spill_delta,
        "active_core_delta": int(chain.get("active_core_count", 0)) - int(
            baseline.get("active_core_count", 0)
        ),
        "max_m_compute_delta": chain_m - baseline_m,
        "max_v_compute_delta": chain_v - baseline_v,
        "m_load_imbalance_delta": float(chain.get("m_load_imbalance", 0.0)) - float(
            baseline.get("m_load_imbalance", 0.0)
        ),
        "v_load_imbalance_delta": float(chain.get("v_load_imbalance", 0.0)) - float(
            baseline.get("v_load_imbalance", 0.0)
        ),
        "actual_group_count_delta": int(chain.get("actual_group_count", 0)) - int(
            baseline.get("actual_group_count", 0)
        ),
        "baseline_plan_hash": baseline.get("plan_hash"),
        "chain_plan_hash": chain.get("plan_hash"),
        "classification": classification,
    }


def analyze(rows: list[dict[str, Any]], baseline_arm: str, chain_arm: str) -> dict[str, Any]:
    baseline = _select_rows(rows, baseline_arm)
    chain = _select_rows(rows, chain_arm)
    comparisons = [
        _comparison(baseline[key], chain[key])
        for key in sorted(set(baseline) & set(chain))
    ]
    classifications = Counter(item["classification"] for item in comparisons)
    return {
        "baseline_arm": baseline_arm,
        "chain_arm": chain_arm,
        "comparison_count": len(comparisons),
        "missing_baseline": len(set(chain) - set(baseline)),
        "missing_chain": len(set(baseline) - set(chain)),
        "wins": sum(item["makespan_delta"] < 0 for item in comparisons),
        "ties": sum(item["makespan_delta"] == 0 for item in comparisons),
        "losses": sum(item["makespan_delta"] > 0 for item in comparisons),
        "classification_counts": dict(sorted(classifications.items())),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze chain vs contiguous official results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-arm", default="contiguous_p2")
    parser.add_argument("--chain-arm", default="chain_contiguous")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    summary_path = results_dir / "phase4_matrix.json"
    if not summary_path.is_file():
        parser.error(f"missing phase4_matrix.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = analyze(summary.get("comparisons", []), args.baseline_arm, args.chain_arm)
    result["matrix_spec"] = summary.get("matrix_spec", {})
    _write_json(output_dir / "chain_attribution.json", result)
    if result["comparisons"]:
        fields = list(result["comparisons"][0])
        with (output_dir / "chain_attribution.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["comparisons"])
    print(
        f"comparisons={result['comparison_count']} wins={result['wins']} "
        f"ties={result['ties']} losses={result['losses']} "
        f"output={output_dir / 'chain_attribution.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())