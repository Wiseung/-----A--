from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from solver.evaluate_adapter import EvaluationError, EvaluationResult
from solver.graph_analysis import analyze_graph
from solver.graph_io import load_config, parse_graph
from solver.schedule_b import generate_schedule as schedule_2
from solver.run_identity import build_fingerprints
from solver.solver import _store_summary, plan_signature, solve_one
from tests.helpers import make_chain_graph


class StubEvaluator:
    def __init__(self, timeout_sec: float, fail_multicore: bool = False) -> None:
        self.timeout_sec = timeout_sec
        self.calls = 0
        self.fail_multicore = fail_multicore
        self.multicore_timeout: float | None = None
        self.plans: list[dict] = []

    def evaluate_singlecore(self, graph_path: Path, tag: str) -> EvaluationResult:
        self.calls += 1
        time.sleep(0.02)
        return EvaluationResult({"makespan": 100}, 0.02, "")

    def evaluate_multicore(
        self, graph_path: Path, plan: dict, problem: int, tag: str
    ) -> EvaluationResult:
        self.calls += 1
        self.multicore_timeout = self.timeout_sec
        self.plans.append(plan)
        time.sleep(0.02)
        if self.fail_multicore:
            raise EvaluationError("synthetic failure", 0.02)
        return EvaluationResult(
            {
                "makespan": 20,
                "data_movement_bytes": {
                    "added_copy_bytes": 0,
                    "original_graph_copy_bytes": 0,
                    "scheduled_copy_bytes": 0,
                },
                "cache_stats": {"hit_rate": 0.0},
            },
            0.02,
            "",
        )


class SolverReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(cls.root / "2026_official")
        cls.graph = parse_graph(make_chain_graph(2), cls.root / "case_test.json")

    def _solve(
        self,
        results_dir: Path,
        evaluator: StubEvaluator,
        problem: int = 2,
        ncores: int = 2,
        max_evals: int = 1,
        improve: bool = False,
    ) -> dict:
        return solve_one(
            self.graph,
            problem,
            ncores,
            self.config,
            evaluator,
            results_dir,
            time_limit=2.0,
            max_evals=max_evals,
            seed=0,
            improve=improve,
            baseline_timeout=1.0,
            retry_baseline=True,
            experiment_id="test",
            run_id="test-run",
            history_dir=results_dir,
        )

    def test_timing_separates_baseline_search_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = StubEvaluator(timeout_sec=0.001)
            row = self._solve(Path(temporary), evaluator)

        self.assertEqual(row["status"], "evaluated")
        self.assertGreater(row["baseline_wall_sec"], 0)
        self.assertGreater(row["search_wall_sec"], 0)
        self.assertAlmostEqual(
            row["solver_runtime_sec"],
            row["search_wall_sec"] - row["candidate_eval_wall_sec"],
            delta=1e-9,
        )
        self.assertGreaterEqual(
            row["total_wall_sec"],
            row["baseline_wall_sec"] + row["search_wall_sec"] - 0.01,
        )
        self.assertLessEqual(evaluator.multicore_timeout, 0.001)
        self.assertEqual(evaluator.timeout_sec, 0.001)

    def test_failed_evaluation_does_not_replace_validated_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            best_path = (
                results_dir / "schedules" /
                "case_test_p2_n2_test-run_best_validated.json"
            )
            best_path.parent.mkdir(parents=True)
            previous_plan = schedule_2(
                analyze_graph(self.graph), 2, self.config, 2
            )
            best_path.write_text(
                json.dumps(previous_plan, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            previous_bytes = best_path.read_bytes()

            row = self._solve(
                results_dir,
                StubEvaluator(timeout_sec=1.0, fail_multicore=True),
            )

            self.assertEqual(row["status"], "evaluation_failed")
            self.assertEqual(best_path.read_bytes(), previous_bytes)
            self.assertTrue((results_dir / "schedules" / "case_test_p2_n2_test-run_legal_draft.json").is_file())

    def test_problem_3_tries_problem_2_candidate_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            evaluator = StubEvaluator(timeout_sec=1.0)
            row = self._solve(results_dir, evaluator, problem=3)

            expected_plan = schedule_2(
                analyze_graph(self.graph), 2, self.config, 2
            )
            self.assertEqual(len(evaluator.plans), 1)
            self.assertEqual(evaluator.plans[0], expected_plan)
            self.assertTrue((results_dir / "candidate_trials.jsonl").is_file())
            trial = json.loads(
                (results_dir / "candidate_trials.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertTrue(trial["candidate_family"].startswith("problem2_warm_start_g"))
            self.assertEqual(
                row["best_candidate_source"], trial["candidate_family"]
            )
            validated_dir = results_dir / "validated"
            schedules_dir = validated_dir / "schedules"
            schedules_dir.mkdir(parents=True)
            saved_plan = schedule_2(
                analyze_graph(self.graph), 2, self.config, 1
            )
            plan_path = schedules_dir / (
                "case_test_p2_n2_test-run_best_validated.json"
            )
            plan_path.write_text(
                json.dumps(saved_plan, ensure_ascii=False), encoding="utf-8"
            )
            fingerprints = build_fingerprints(
                self.graph.path,
                self.config["path"],
                self.root / "2026_official",
                self.root / "solver",
            )
            metadata = {
                "problem": 2,
                "ncores": 2,
                "plan_hash": hashlib.sha256(
                    plan_signature(saved_plan).encode("utf-8")
                ).hexdigest(),
                **fingerprints,
            }
            plan_path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )
            saved_evaluator = StubEvaluator(timeout_sec=1.0)
            saved_evaluator.config_path = self.config["path"]

            saved_row = self._solve(validated_dir, saved_evaluator, problem=3)

            self.assertEqual(saved_evaluator.plans[0], saved_plan)
            self.assertEqual(
                saved_row["best_candidate_source"],
                "problem2_validated_warm_start",
            )
            saved_trial = json.loads(
                (validated_dir / "candidate_trials.jsonl")
                .read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                saved_trial["candidate_family"],
                "problem2_validated_warm_start",
            )
            self.assertEqual(saved_trial["source_plan"], plan_path.as_posix())

    def test_lower_core_plan_is_extended_with_empty_core(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            previous_plan = schedule_2(
                analyze_graph(self.graph), 1, self.config, 1
            )
            prior_path = (
                results_dir / "schedules" / "case_test_p2_n1_best.json"
            )
            prior_path.parent.mkdir(parents=True)
            prior_path.write_text(
                json.dumps(previous_plan, ensure_ascii=False), encoding="utf-8"
            )
            evaluator = StubEvaluator(timeout_sec=1.0)

            self._solve(
                results_dir,
                evaluator,
                ncores=2,
                max_evals=2,
            )

            self.assertEqual(len(evaluator.plans), 2)
            self.assertEqual(
                evaluator.plans[1]["core_schedules"],
                [previous_plan["core_schedules"][0], []],
            )
            trials = [
                json.loads(line)
                for line in (results_dir / "candidate_trials.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                trials[1]["candidate_family"],
                "previous_core_count_warm_start",
            )

    def test_solver_records_local_neighbor_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            row = self._solve(
                results_dir,
                StubEvaluator(timeout_sec=1.0),
                max_evals=3,
                improve=True,
            )
            trials = [
                json.loads(line)
                for line in (results_dir / "candidate_trials.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(row["status"], "evaluated")
        self.assertTrue(any(
            trial["candidate_family"].startswith("local") for trial in trials
        ))

    def test_summary_keeps_distinct_run_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            base = {
                "experiment_id": "test",
                "run_id": "run-a",
                "case": "case_test",
                "problem": 2,
                "ncores": 2,
                "graph_hash": "graph",
                "config_hash": "config",
                "official_code_hash": "official",
                "solver_code_hash": "solver",
                "method": "method",
                "seed": 1,
                "time_limit_sec": 1.0,
                "baseline_timeout_sec": 1.0,
                "evaluator_timeout_sec": 1.0,
                "max_evals": 1,
                "improve_enabled": False,
                "status": "evaluated",
            }
            _store_summary(results_dir, base)
            _store_summary(results_dir, {**base, "run_id": "run-b"})

            stored = json.loads(
                (results_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual([row["run_id"] for row in stored], ["run-a", "run-b"])


if __name__ == "__main__":
    unittest.main()
