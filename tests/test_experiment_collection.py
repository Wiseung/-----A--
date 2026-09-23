from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.collect_results import collect


class ResultCollectionTests(unittest.TestCase):
    def test_manifest_denominator_includes_unobserved_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            cases = [f"case_{index:03d}" for index in range(1, 101)]
            (results_dir / "batch_manifest.json").write_text(
                json.dumps({
                    "cases": cases,
                    "problems": [1, 2, 3],
                    "ncores": [2, 3, 4, 5],
                    "data_dir": "synthetic",
                }),
                encoding="utf-8",
            )
            summary = [{
                "experiment_id": "test",
                "run_id": "run-1",
                "case": "case_001",
                "problem": 2,
                "ncores": 4,
                "method": "synthetic",
                "graph_hash": "graph",
                "config_hash": "config",
                "official_code_hash": "official",
                "solver_code_hash": "solver",
                "time_limit_sec": 1.0,
                "baseline_timeout_sec": 1.0,
                "evaluator_timeout_sec": 1.0,
                "max_evals": 1,
                "improve_enabled": False,
                "retry_baseline": False,
                "single_core_baseline": 100,
                "makespan": 50,
                "status": "evaluated",
            }]
            summary_path = results_dir / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            original_summary = summary_path.read_bytes()

            rows, aggregate = collect(results_dir)

            complete = json.loads(
                (results_dir / "complete_case_manifest.json").read_text(encoding="utf-8")
            )
            run = complete["runs"][0]
            p2_n4 = next(
                row for row in aggregate
                if row["experiment_id"] == "test"
                and row["run_id"] == "run-1"
                and row["problem"] == 2
                and row["ncores"] == 4
                and row["series"] == "no_l2"
            )
            summary_preserved = summary_path.read_bytes() == original_summary

        self.assertEqual(len(rows), 1)
        self.assertEqual(run["planned_combinations"], 1200)
        self.assertEqual(run["main_multicore_planned_combinations"], 1200)
        self.assertEqual(run["success_combinations"], 1)
        self.assertEqual(run["not_run_combinations"], 1199)
        self.assertEqual(p2_n4["planned_count"], 100)
        self.assertEqual(p2_n4["case_count"], 1)
        self.assertEqual(p2_n4["missing_cases"], 99)
        self.assertEqual(p2_n4["mean_speedup"], 2.0)
        self.assertTrue(summary_preserved)


if __name__ == "__main__":
    unittest.main()
