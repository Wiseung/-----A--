from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.collect_results import collect
from experiments.phase4_stratified_matrix import (
    ARMS,
    _candidate_matrix,
    _reserve_statistics,
    _rank_proxy_candidates,
)
from experiments.evaluate_selected_plans import _select_rows
from experiments.fixed_plan_cache_comparison import (
    _plan_hash,
    _strict_pair_summary,
)
from experiments.promote_phase4_candidates import evaluate_promotion
from experiments.summarize_chain_matrix import _strict_pair_statistics


class ResultCollectionTests(unittest.TestCase):
    def test_problem3_selection_keeps_both_arms_for_same_hash(self) -> None:
        rows = [
            {
                "case": "case_001",
                "ncores": 2,
                "arm": "contiguous_p2",
                "group_count": 8,
                "plan_hash": "shared",
                "evaluation_problem": 2,
                "status": "success",
                "makespan": 100,
            },
            {
                "case": "case_001",
                "ncores": 2,
                "arm": "chain_contiguous",
                "group_count": 10,
                "plan_hash": "shared",
                "evaluation_problem": 2,
                "status": "equivalent_plan",
                "makespan": 100,
            },
        ]

        selected = _select_rows(
            rows, 1, {"contiguous_p2", "chain_contiguous"}
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual({row["arm"] for row in selected}, {
            "contiguous_p2", "chain_contiguous"
        })
        self.assertEqual({row["plan_hash"] for row in selected}, {"shared"})

    def test_fixed_plan_cache_pair_requires_matching_hash(self) -> None:
        plan = {
            "node_to_subgraph": {"20": 0},
            "core_schedules": [[0], []],
        }
        plan_hash = _plan_hash(plan)
        pair = _strict_pair_summary(
            2,
            {"status": "evaluated", "plan_hash": plan_hash, "makespan": 100},
            {"status": "evaluated", "plan_hash": plan_hash, "makespan": 80},
        )

        self.assertTrue(pair["strict_pair_complete"])
        self.assertEqual(pair["p2_plan_hash"], pair["p3_plan_hash"])
        self.assertAlmostEqual(pair["p3_over_p2_makespan"], 0.8)

        mismatched = _strict_pair_summary(
            2,
            {"status": "evaluated", "plan_hash": "p2", "makespan": 100},
            {"status": "evaluated", "plan_hash": "p3", "makespan": 80},
        )

        self.assertFalse(mismatched["strict_pair_complete"])
        self.assertIsNone(mismatched["p3_over_p2_makespan"])

    def test_strict_plan_pairing_marks_missing_arm_and_deduplicates_cache(self) -> None:
        p2_rows = []
        p3_rows = []
        for arm in ("contiguous_p2", "chain_contiguous"):
            p2_rows.append({
                "case": "case_001",
                "ncores": 2,
                "arm": arm,
                "group_count": 8,
                "plan_hash": "shared",
                "status": "success",
                "makespan": 100,
                "added_copy_bytes": 0,
                "spill_added_copy_bytes": 0,
                "candidate_id": f"p2-{arm}",
                "source_results_dir": "p2-source",
            })
            p3_rows.append({
                "case": "case_001",
                "ncores": 2,
                "arm": arm,
                "group_count": 8,
                "plan_hash": "shared",
                "status": "success",
                "makespan_p3": 80,
                "added_copy_bytes_p3": 0,
                "cache_hit_bytes": 40,
                "cache_miss_bytes": 60,
                "candidate_id": f"p3-{arm}",
                "source_results_dir": "p2-source",
            })
            if arm == "chain_contiguous":
                p2_rows.append({
                    "case": "case_001",
                    "ncores": 2,
                    "arm": arm,
                    "group_count": 12,
                    "plan_hash": "chain-only",
                    "status": "success",
                    "makespan": 90,
                    "added_copy_bytes": 0,
                    "spill_added_copy_bytes": 0,
                    "candidate_id": "p2-chain-only",
                })
                p3_rows.append({
                    "case": "case_001",
                    "ncores": 2,
                    "arm": arm,
                    "group_count": 12,
                    "plan_hash": "chain-only",
                    "status": "success",
                    "makespan_p3": 70,
                    "added_copy_bytes_p3": 0,
                    "cache_hit_bytes": 30,
                    "cache_miss_bytes": 70,
                    "candidate_id": "p3-chain-only",
                })

        result = _strict_pair_statistics(p2_rows, p3_rows)
        chain_only = next(
            row for row in result["p3_pair_rows"]
            if row["plan_hash"] == "chain-only"
        )

        self.assertEqual(result["complete_p2_p3_pair_count"], 3)
        self.assertEqual(result["p3_only_chain_plan_count"], 1)
        self.assertEqual(result["cache_hit_bytes_total"], 70)
        self.assertEqual(result["cache_miss_bytes_total"], 130)
        self.assertAlmostEqual(result["cache_hit_rate_weighted"], 0.35)
        self.assertEqual(chain_only["p3_same_hash_missing_arm"], "contiguous_p2")

    def test_high_risk_reserve_statistics(self) -> None:
        candidates = [
            {
                "candidate_id": "chain-high",
                "candidate_status": "proxy_selected_high_risk",
            },
            {
                "candidate_id": "chain-safe",
                "candidate_status": "proxy_selected",
            },
        ]
        comparisons = [
            {
                "candidate_id": "chain-high",
                "case": "case_028",
                "ncores": 4,
                "arm": "chain_contiguous",
                "evaluation_problem": 2,
                "status": "success",
                "makespan": 90,
                "added_copy_bytes": 10,
            },
            {
                "candidate_id": "base",
                "case": "case_028",
                "ncores": 4,
                "arm": "contiguous_p2",
                "evaluation_problem": 2,
                "status": "success",
                "makespan": 100,
                "added_copy_bytes": 10,
            },
        ]

        stats = _reserve_statistics(candidates, comparisons)

        self.assertEqual(stats["high_risk_reserved_count"], 1)
        self.assertEqual(stats["high_risk_evaluated_count"], 1)
        self.assertEqual(stats["high_risk_wins"], 1)
        self.assertEqual(stats["high_risk_recovery_rate"], 1.0)

    def test_phase5_promotion_gates_select_non_regressing_arm(self) -> None:
        matrix_spec = {
            "cases": ["case_001", "case_014"],
            "ncores": [2, 4],
            "evaluation_problems": [2, 3],
        }
        comparisons = []
        for case in matrix_spec["cases"]:
            for ncores in matrix_spec["ncores"]:
                for problem in matrix_spec["evaluation_problems"]:
                    baseline_arm = "contiguous_p2" if problem == 2 else "contiguous_p3"
                    for arm, makespan in (
                        (baseline_arm, 100),
                        ("cagg_lite", 90),
                        ("cagg_lite_coverage", 110),
                    ):
                        comparisons.append({
                            "arm": arm,
                            "case": case,
                            "ncores": ncores,
                            "evaluation_problem": problem,
                            "group_count": 8,
                            "status": "success",
                            "makespan": makespan,
                            "added_copy_bytes": makespan * 10,
                            "spill_added_copy_bytes": makespan * 5,
                            "active_core_count": ncores,
                        })

        result = evaluate_promotion({
            "matrix_spec": matrix_spec,
            "comparisons": comparisons,
        }, min_wins=1, min_win_cases=2)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["recommended_arm"], "cagg_lite")
        cagg = next(item for item in result["candidates"] if item["arm"] == "cagg_lite")
        self.assertEqual((cagg["wins"], cagg["ties"], cagg["losses"]), (8, 0, 0))
        self.assertTrue(all(cagg["gates"].values()))

    def test_phase5_marks_missing_matrix_as_not_ready(self) -> None:
        result = evaluate_promotion({
            "matrix_spec": {
                "cases": ["case_001"],
                "ncores": [2],
                "evaluation_problems": [2, 3],
            },
            "comparisons": [],
        })

        self.assertEqual(result["status"], "not_ready")
        self.assertIsNone(result["recommended_arm"])

    def test_phase4_matrix_and_proxy_selection(self) -> None:
        candidates = _candidate_matrix(
            ["case_001"], [2], [8, 10, 12], list(ARMS), "run-1"
        )
        for candidate in candidates:
            candidate["proxy_score"] = (
                float(candidate["group_count"]), 0.0, 0.0, 0, 0, 0
            )

        _rank_proxy_candidates(candidates, proxy_screen=True)

        self.assertEqual(len(candidates), len(ARMS) * 3)
        self.assertEqual(
            sum(candidate["proxy_selected"] for candidate in candidates),
            len(ARMS),
        )
        self.assertEqual(
            sum(candidate["candidate_status"] == "screened_out" for candidate in candidates),
            len(ARMS) * 2,
        )

    def test_chain_safety_gate_filters_boundary_explosion(self) -> None:
        candidates = _candidate_matrix(
            ["case_028"], [4], [8], ["contiguous_p2", "chain_contiguous"], "run-1"
        )
        for candidate in candidates:
            candidate["proxy_score"] = (100.0, 100.0, 10.0, 100.0, 0.0, 0, -4)
            candidate["boundary_copy_bytes_est"] = (
                100 if candidate["arm"] == "contiguous_p2" else 250
            )
            candidate["max_l1_overflow_bytes_est"] = 10
            candidate["max_ub_overflow_bytes_est"] = 0

        _rank_proxy_candidates(candidates, proxy_screen=True)

        chain = next(item for item in candidates if item["arm"] == "chain_contiguous")
        self.assertEqual(chain["safety_status"], "high_risk")
        self.assertTrue(chain["proxy_selected"])
        self.assertEqual(chain["candidate_status"], "proxy_selected_high_risk")
        self.assertEqual(chain["proxy_selection_reasons"], ["high_risk_reserve"])

        unfiltered = [dict(candidate) for candidate in candidates]
        _rank_proxy_candidates(unfiltered, proxy_screen=False)
        self.assertTrue(all(candidate["proxy_selected"] for candidate in unfiltered))

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
