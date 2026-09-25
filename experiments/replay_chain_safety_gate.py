from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str, int]:
    return (
        row["case"],
        row["ncores"],
        row["group_count"],
        row["arm"],
        row["evaluation_problem"],
    )


def replay(
    official_rows: list[dict[str, Any]],
    proxy_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    official = {
        _row_key(row): row
        for row in official_rows
        if row.get("status") in {"success", "equivalent_plan"}
    }
    proxy = {
        (
            row["case"], row["ncores"], row["group_count"], row["arm"], 2
        ): row
        for row in proxy_candidates
        if row.get("arm") == "chain_contiguous"
    }
    comparisons = []
    for key, chain in sorted(proxy.items()):
        case, ncores, group_count, arm, problem = key
        if chain.get("generation_status") != "generated":
            continue
        chain_official = official.get(key)
        baseline = official.get((case, ncores, group_count, "contiguous_p2", problem))
        if chain_official is None or baseline is None:
            continue
        chain_makespan = float(chain_official["makespan"])
        baseline_makespan = float(baseline["makespan"])
        outcome = (
            "win" if chain_makespan < baseline_makespan
            else "tie" if chain_makespan == baseline_makespan
            else "loss"
        )
        safety_status = chain.get("safety_status", "unknown")
        gate_filtered = safety_status == "high_risk"
        if gate_filtered and outcome == "win":
            gate_result = "false_negative"
        elif gate_filtered and outcome == "loss":
            gate_result = "correctly_filtered"
        elif not gate_filtered and outcome == "win":
            gate_result = "correctly_retained"
        elif not gate_filtered and outcome == "loss":
            gate_result = "leaked_regression"
        else:
            gate_result = "tie"
        comparisons.append({
            "case": case,
            "ncores": ncores,
            "group_count": group_count,
            "arm": arm,
            "official_outcome": outcome,
            "gate_result": gate_result,
            "safety_status": safety_status,
            "gate_decision": "filter" if gate_filtered else "retain",
            "gate_reason": (
                "boundary_ratio_and_overflow"
                if gate_filtered else "not_high_risk"
            ),
            "false_negative_risk": gate_result == "false_negative",
            "boundary_ratio_to_contiguous": chain.get(
                "boundary_ratio_to_contiguous"
            ),
            "overflow_delta_to_contiguous": chain.get(
                "overflow_delta_to_contiguous"
            ),
            "chain_makespan": chain_official.get("makespan"),
            "baseline_makespan": baseline.get("makespan"),
            "makespan_delta": chain_makespan - baseline_makespan,
            "chain_partition_copy_bytes": chain_official.get(
                "partition_added_copy_bytes"
            ),
            "baseline_partition_copy_bytes": baseline.get(
                "partition_added_copy_bytes"
            ),
            "chain_spill_added_copy_bytes": chain_official.get(
                "spill_added_copy_bytes"
            ),
            "baseline_spill_added_copy_bytes": baseline.get(
                "spill_added_copy_bytes"
            ),
            "chain_plan_hash": chain_official.get("plan_hash"),
            "baseline_plan_hash": baseline.get("plan_hash"),
        })
    counts = Counter(item["gate_result"] for item in comparisons)
    return {
        "comparison_count": len(comparisons),
        "gate_result_counts": dict(sorted(counts.items())),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay chain safety gate on official results.")
    parser.add_argument("--official-results-dir", type=Path, required=True)
    parser.add_argument("--proxy-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    official_dir = args.official_results_dir.resolve()
    proxy_dir = args.proxy_results_dir.resolve()
    output_dir = (args.output_dir or official_dir).resolve()
    official_path = official_dir / "phase4_matrix.json"
    proxy_path = proxy_dir / "subgraph_diagnostics.json"
    if not official_path.is_file() or not proxy_path.is_file():
        parser.error("both phase4_matrix.json and subgraph_diagnostics.json are required")
    official = json.loads(official_path.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    result = replay(official.get("comparisons", []), proxy.get("candidates", []))
    result["official_results_dir"] = str(official_dir)
    result["proxy_results_dir"] = str(proxy_dir)
    _write_json(output_dir / "chain_safety_replay.json", result)
    if result["comparisons"]:
        fields = list(result["comparisons"][0])
        with (output_dir / "chain_safety_replay.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["comparisons"])
    print(
        f"comparisons={result['comparison_count']} "
        f"counts={result['gate_result_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())