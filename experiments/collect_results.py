from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def formal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("case", "").startswith("case_")]


def collect(results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = results_dir / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
    rows = formal_rows(rows)
    by_key = {(row["case"], row["problem"], row["ncores"]): row for row in rows}

    for row in rows:
        if row.get("problem") == 3:
            no_l2 = by_key.get((row["case"], 2, row["ncores"]))
            row["makespan_no_l2"] = no_l2.get("makespan") if no_l2 else None
            row["makespan_l2"] = row.get("makespan")
            if row["makespan_no_l2"] and row["makespan_l2"]:
                row["cache_speedup"] = row["makespan_no_l2"] / row["makespan_l2"]
            else:
                row["cache_speedup"] = None

    aggregate = []
    cases = sorted({row["case"] for row in rows})
    for problem in (1, 2):
        for ncores in range(1, 6):
            speedups = []
            missing = 0
            for case in cases:
                if ncores == 1 and problem == 1:
                    baseline = next((
                        row.get("single_core_baseline") for row in rows
                        if row["case"] == case and row.get("single_core_baseline") is not None
                    ), None)
                    if baseline is None:
                        missing += 1
                    else:
                        speedups.append(1.0)
                    continue
                row = by_key.get((case, problem, ncores))
                if row and row.get("makespan") and row.get("single_core_baseline"):
                    speedups.append(row["single_core_baseline"] / row["makespan"])
                else:
                    missing += 1
            aggregate.append({
                "problem": problem,
                "ncores": ncores,
                "series": "no_l2",
                "case_count": len(speedups),
                "missing_cases": missing,
                "mean_speedup": statistics.mean(speedups) if speedups else None,
            })

    for ncores in range(1, 6):
        for series, problem in (("no_l2", 2), ("l2", 3)):
            speedups = []
            missing = 0
            for case in cases:
                if ncores == 1:
                    source = by_key.get((case, problem, 1))
                    if (source and source.get("makespan")
                            and source.get("single_core_baseline")):
                        speedups.append(
                            source["single_core_baseline"] / source["makespan"]
                        )
                    else:
                        missing += 1
                    continue
                row = by_key.get((case, problem, ncores))
                if row and row.get("makespan") and row.get("single_core_baseline"):
                    speedups.append(row["single_core_baseline"] / row["makespan"])
                else:
                    missing += 1
            aggregate.append({
                "problem": 3,
                "ncores": ncores,
                "series": series,
                "case_count": len(speedups),
                "missing_cases": missing,
                "mean_speedup": statistics.mean(speedups) if speedups else None,
            })

    for ncores in range(1, 6):
        ratios = [
            row["cache_speedup"] for row in rows
            if row.get("problem") == 3 and row.get("ncores") == ncores
            and row.get("cache_speedup") is not None
        ]
        aggregate.append({
            "problem": 3,
            "ncores": ncores,
            "series": "cache_speedup",
            "case_count": len(ratios),
            "missing_cases": max(0, len(cases) - len(ratios)),
            "mean_speedup": statistics.mean(ratios) if ratios else None,
        })

    (results_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (results_dir / "aggregate_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "problem", "ncores", "series", "case_count", "missing_cases", "mean_speedup"
        ])
        writer.writeheader()
        writer.writerows(aggregate)
    with (results_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(rows[0]) if rows else ["case", "problem", "ncores"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows, aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute arithmetic mean speedups from evaluator results.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, aggregate = collect(args.results_dir)
    print(f"cases={len({row['case'] for row in rows})}; result_rows={len(rows)}")
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